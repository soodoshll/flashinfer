# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This file is ported from CUTLASS's Rubin dense_blockscaled_gemm_persistent.py
# with modifications for FlashInfer integration (alpha scaling, tensor-based API, wrapper method).

from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.gemm.sm100 as epilogue_sm100
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor
    tSF_compact: cute.Tensor


class Sm107BlockScaledPersistentDenseGemmKernel:
    """Persistent dense block scaled GEMM kernel for Rubin SM107.

    Implements batched matrix multiplication (C = alpha * A x SFA x B x SFB)
    with support for various data types and architectural features specific
    to Rubin GPUs with persistent tile scheduling and warp specialization.

    Args:
        sf_vec_size: Scalefactor vector size.
        mma_inst_shape: Shape of the MMA instruction (M, N, K).
        mma_tiler: Shape of the MMA tiler (M, N, K).
        cluster_shape_mn: Cluster dimensions (M, N) for parallel processing.

    Notes:
        Supported combinations of A/B data types, SF data types and SF vector size:
            - MXF8: A/B: Float8E5M2/Float8E4M3FN + SF: Float8E8M0FNU + sf_vec_size: 32
            - NVF4: A/B: Float4E2M1FN + SF: Float8E8M0FNU/Float8E4M3FN/FloatNV8E5M3FNU + sf_vec_size: 16/32
        Supported accumulator data types: Float32
        Supported C data types: Float32, Float16/BFloat16
        Constraints:
            - Mma tiler M must be 128, 256 or 512, MMA instruction shape M can be 128 or 256
            - Mma tiler N and MMA instruction shape N must be 64/128/192/256
            - B-reuse feature is enabled if (MMA tiler M // MMA instruction shape M) == 2
            - Cluster shape M must be multiple of 2 if Mma instruction shape M is 256 (.2CTA)
            - Cluster shape M/N must be positive and power of 2, total cluster size <= 16
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        self.acc_dtype = cutlass.Float32
        self.sf_vec_size = sf_vec_size
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cluster_shape_mn = cluster_shape_mn
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.arch = "sm_107"
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id)
        )

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp
            * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.epilogue_warp_id = self.epilog_warp_id
        self.epilog_sync_bar_id = self.epilog_sync_barrier.barrier_id

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.enable_breuse = True if mma_tiler[0] // mma_inst_shape[0] == 2 else False

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        c_dtype: Type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        smem_capacity: int,
        occupancy: int,
        with_breuse: bool,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics."""
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2
        num_c_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1,
        )
        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, 1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        num_ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        num_c_stage += (
            smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)

        return num_acc_stage, num_ab_stage, num_c_stage

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs."""
        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            self.cta_group, self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.epi_tile = sm107_utils.compute_epilogue_tile_shape(
            tiled_mma.op,
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.c_layout,
            self.c_dtype,
        )
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.a_dtype, self.b_dtype,
            self.epi_tile, self.c_dtype, self.c_layout,
            self.sf_dtype, self.sf_vec_size,
            self.smem_capacity, self.occupancy, self.enable_breuse,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage,
        )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage,
        )

        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        self.num_sfa_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFA_layout))
            & 0x0000FFFF
        )
        self.num_sfb_tmem_cols = (
            cute.cosize(cute.recast_layout(32, self.sf_dtype.width, self.tCtSFB_layout))
            & 0x0000FFFF
        )
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1]
            * self.num_acc_stage
            * (2 if self.enable_breuse else 1)
        )

    def _is_interleaved_utccp(self) -> bool:
        return (
            self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse
        )

    def _mainloop_s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor,
        then partition smem (source) and tensor memory (destination)."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # Workaround for vector size 16: add broadcasting mode in the
        # pre-partitioned tensor to produce a suitable partitioned layout
        # for the TMEM destination.
        def appendMNBroadcastMode(smem_layout: cute.Layout):
            mn_dim = cute.get(smem_layout, mode=[0, 0])
            mn_dim = cute.append(mn_dim, cute.make_layout((4), stride=(0)))
            layout = cute.append(
                cute.group_modes(mn_dim, 0), cute.get(smem_layout, mode=[0, 1])
            )
            layout = cute.append(
                cute.group_modes(layout, 0), cute.get(smem_layout, mode=[1])
            )
            layout = cute.append(layout, cute.get(smem_layout, mode=[2]))
            layout = cute.append(layout, cute.get(smem_layout, mode=[3]))
            return layout

        tCsSF_compact_bcast = cute.make_tensor(
            tCsSF_compact.iterator, appendMNBroadcastMode(tCsSF_compact.layout)
        )

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact_bcast)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return S2TCopyBundle(tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t)

    def _mainloop_s2t_copies(
        self, stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle, sfb_s2t_bundle: S2TCopyBundle,
    ):
        s2t_stage_coord = (None, None, None, None, stage_idx)
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfa_s2t_bundle.tSF_compact,
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s2t_stage_coord],
            sfb_s2t_bundle.tSF_compact,
        )

    def _mainloop_s2t_interleaved_copies(
        self, k_block: int, stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle, sfb_s2t_bundle: S2TCopyBundle,
    ):
        s_sfa_crd_keep = (None, 0, None, k_block, stage_idx)
        s_sfa_crd_reuse = (None, 1, None, k_block, stage_idx)
        s_sfb_crd = (None, None, None, k_block, stage_idx)
        t_sfa_crd_keep = (None, 0, None, k_block)
        t_sfa_crd_reuse = (None, 1, None, k_block)
        t_sfb_crd = (None, None, None, k_block)

        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_keep],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_keep],
        )
        cute.copy(
            sfb_s2t_bundle.tiled_copy,
            sfb_s2t_bundle.sSF_compact[s_sfb_crd],
            sfb_s2t_bundle.tSF_compact[t_sfb_crd],
        )
        cute.copy(
            sfa_s2t_bundle.tiled_copy,
            sfa_s2t_bundle.sSF_compact[s_sfa_crd_reuse],
            sfa_s2t_bundle.tSF_compact[t_sfa_crd_reuse],
        )

    @cute.jit
    def __call__(
        self, a_tensor: cute.Tensor, b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor, sfb_tensor: cute.Tensor,
        c_tensor: cute.Tensor, alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the GEMM operation."""
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a_tensor).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b_tensor).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c_tensor)

        self._setup_attributes()

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, self.sf_vec_size)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, self.sf_vec_size)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
            self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
                self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype, self.b_dtype, self.a_major_mode, self.b_major_mode,
                self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD, b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk, permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )

        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb_tensor, sfb_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16,
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)
            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1], tma_tensor_sfb.shape[2],
            )
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1], tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, cute.make_layout(new_shape, stride=new_stride)
            )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), c_tensor, epi_smem_layout, self.epi_tile,
        )

        self.tile_sched_params, grid = self._compute_grid(
            c_tensor, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters,
        )

        self.buffer_align_bytes = 1024

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)], self.buffer_align_bytes]
            sA: cute.struct.Align[cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)], self.buffer_align_bytes]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)], self.buffer_align_bytes]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)], self.buffer_align_bytes]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)], self.buffer_align_bytes]

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma, tiled_mma_bkeep, tiled_mma_breuse, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_smem_layout_staged,
            self.tCtSFA_layout, self.tCtSFB_layout,
            self.c_smem_layout_staged, self.epi_tile, self.tile_sched_params,
            epilogue_op, alpha,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1), stream=stream, min_blocks_per_mp=1,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom, mC_mnl: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout, tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
        alpha: cute.Tensor,
    ):
        """GPU device kernel performing the Persistent batched GEMM computation."""
        alpha_value = alpha[0].to(cutlass.Float32)

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        sfa_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
            sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        gA_mkl = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA_mkl = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB_nkl = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3),
        )

        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # =====================================================================
        # TMA load warp
        # =====================================================================
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1], cur_tile_coord[2],
                )

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]

                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                    cute.copy(tma_atom_a, tAgA_slice[(None, ab_producer_state.count)], tAsA[(None, ab_producer_state.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=a_full_mcast_mask)
                    cute.copy(tma_atom_b, tBgB_slice[(None, ab_producer_state.count)], tBsB[(None, ab_producer_state.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=b_full_mcast_mask)
                    cute.copy(tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)], tAsSFA[(None, ab_producer_state.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=sfa_full_mcast_mask)
                    cute.copy(tma_atom_sfb, tBgSFB_slice[(None, ab_producer_state.count)], tBsSFB[(None, ab_producer_state.index)],
                              tma_bar_ptr=ab_pipeline.producer_get_barrier(ab_producer_state), mcast_mask=sfb_full_mcast_mask)

                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_pipeline.producer_tail(ab_producer_state)

        # =====================================================================
        # MMA warp
        # =====================================================================
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype)
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1], cur_tile_coord[2],
                )

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] in {64, 192}):
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            self._mainloop_s2t_copies(ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        for k_block in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (None, 0, k_block, ab_consumer_state.index)
                                a_kblk_crd_reuse = (None, 1, k_block, ab_consumer_state.index)
                                b_kblk_crd = (None, 0, k_block, ab_consumer_state.index)
                                sfa_kblk_crd_keep = (None, 0, k_block)
                                sfa_kblk_crd_reuse = (None, 1, k_block)
                                sfb_kblk_crd = (None, 0, k_block)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        k_block, ab_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle,
                                    )

                                tiled_mma_bkeep.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma_bkeep, tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )

                                tiled_mma_breuse.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma_breuse, tCtAcc_breuse,
                                    [tCrA[a_kblk_crd_reuse], tCtSFA[sfa_kblk_crd_reuse]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblk_crd = (None, None, k_block, ab_consumer_state.index)
                                sf_kblk_crd = (None, None, k_block)
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0 or k_block != 0)
                                cute.gemm(
                                    tiled_mma, tCtAcc,
                                    [tCrA[kblk_crd], tCtSFA[sf_kblk_crd]],
                                    [tCrB[kblk_crd], tCtSFB_mma[sf_kblk_crd]],
                                    tCtAcc,
                                )

                        ab_pipeline.consumer_release(ab_consumer_state)

                    ab_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # =====================================================================
        # Epilogue warps
        # =====================================================================
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)

            alpha_epilogue_op = lambda x: epilogue_op(alpha_value * x)

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1], cur_tile_coord[2],
                )

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed
                acc_consumer_state = epilogue_sm100.epilogue_tma_store(
                    self, tidx, warp_idx, tma_atom_c, tCtAcc, sC, tCgC, epi_tile,
                    num_tiles_executed, alpha_epilogue_op,
                    mma_tile_coord_mnl, acc_consumer_state, acc_pipeline, c_pipeline,
                )

            c_pipeline.producer_tail()

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Computes the grid size for the kernel launch."""
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        
        import inspect
        print(f"files: {inspect.getfile(utils.PersistentTileSchedulerParams)}")
        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=True,
        )
        # # Workaround for cutlass-dsl versions where __init__ stores _raster_along_m
        # # (private) but __extract_mlir_values__ reads raster_along_m (public).
        # if not hasattr(tile_sched_params, "raster_along_m") and hasattr(
        #     tile_sched_params, "_raster_along_m"
        # ):
        #     tile_sched_params.raster_along_m = tile_sched_params._raster_along_m
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)

        return tile_sched_params, grid

    @staticmethod
    def can_implement(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        m: int, n: int, k: int, l: int,
        mma_tiler_mn: Tuple[int, int],
        mma_inst_shape: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        a_major: str = "k", b_major: str = "k", c_major: str = "n",
    ) -> bool:
        """Checks if the kernel can implement the given GEMM problem."""
        valid_combinations = {
            (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
            (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 32),
            (cutlass.Float4E2M1FN, cutlass.FloatNV8E5M3FNU, 16),
            (cutlass.Float4E2M1FN, cutlass.FloatNV8E5M3FNU, 32),
            (cutlass.Float8E5M2, cutlass.Float8E8M0FNU, 32),
            (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
        }
        if (ab_dtype, sf_dtype, sf_vec_size) not in valid_combinations:
            return False

        if c_dtype not in {cutlass.Float32, cutlass.Float16, cutlass.BFloat16}:
            return False

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            return False

        if mma_inst_shape[0] not in [128, 256]:
            return False
        if mma_inst_shape[1] not in [64, 128, 192, 256]:
            return False
        if mma_tiler_mn[0] not in [128, 256, 512]:
            return False
        if mma_tiler_mn[1] not in [64, 128, 192, 256]:
            return False

        b_reuse = mma_tiler_mn[0] // mma_inst_shape[0] == 2
        if mma_tiler_mn[0] != mma_inst_shape[0] and not b_reuse:
            return False
        if mma_tiler_mn[1] != mma_inst_shape[1]:
            return False

        if ab_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2}:
            if mma_inst_shape[2] != 64:
                return False
        else:
            if mma_inst_shape[2] != 128:
                return False

        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            return False

        _is_power_of_2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0 or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4 or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        def check_contigous_16B_alignment(dtype, is_mode0_major, tensor_shape):
            major_mode_idx = 0 if is_mode0_major else 1
            num_major_elements = tensor_shape[major_mode_idx]
            num_contiguous_elements = 16 * 8 // dtype.width
            return num_major_elements % num_contiguous_elements == 0

        if (
            not check_contigous_16B_alignment(ab_dtype, a_major == "m", (m, k, l))
            or not check_contigous_16B_alignment(ab_dtype, b_major == "n", (n, k, l))
            or not check_contigous_16B_alignment(c_dtype, c_major == "m", (m, n, l))
        ):
            return False

        return True

    @cute.jit
    def wrapper(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor,
        sf_m: cutlass.Int64, sf_n: cutlass.Int64, sf_k: cutlass.Int64,
        l: cutlass.Constexpr, a_sf_ptr: cute.Pointer, b_sf_ptr: cute.Pointer,
        alpha_tensor: cute.Tensor, max_active_clusters: cutlass.Constexpr,
        current_stream, swap_ab: cutlass.Constexpr = False,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Executes the wrapped GEMM kernel with dynamically shaped tensors.

        Uses TVM-FFI for efficient tensor passing: A, B, C, and alpha are passed
        as cute.Tensor directly. Scale factor tensors remain as pointers because
        their 6D BlockScaledBasicChunk layout cannot be expressed as a torch tensor.
        """
        m = cute.size(mA, mode=[0])
        k_packed = cute.size(mA, mode=[1])
        n = cute.size(mB, mode=[0])
        k = k_packed * 2

        a_fp4_ptr = cute.recast_ptr(mA.iterator, dtype=cutlass.Float4E2M1FN)
        a_tensor = cute.make_tensor(
            a_fp4_ptr, layout=cute.make_ordered_layout((m, k, l), order=(1, 0, 2)),
        )
        b_fp4_ptr = cute.recast_ptr(mB.iterator, dtype=cutlass.Float4E2M1FN)
        b_tensor = cute.make_tensor(
            b_fp4_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2)),
        )
        if cutlass.const_expr(swap_ab):
            c_tensor = cute.make_tensor(
                mC.iterator, layout=cute.make_ordered_layout((m, n, l), order=(0, 1, 2)),
            )
        else:
            c_tensor = cute.make_tensor(
                mC.iterator, layout=cute.make_ordered_layout((m, n, l), order=(1, 0, 2)),
            )
        sfa_tensor = cute.make_tensor(
            a_sf_ptr, layout=cute.make_ordered_layout((32, 4, sf_m, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )
        sfb_tensor = cute.make_tensor(
            b_sf_ptr, layout=cute.make_ordered_layout((32, 4, sf_n, 4, sf_k, l), order=(2, 1, 4, 0, 3, 5)),
        )

        self(
            a_tensor, b_tensor, sfa_tensor, sfb_tensor, c_tensor,
            alpha_tensor, max_active_clusters, current_stream, epilogue_op,
        )
