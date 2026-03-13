# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

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

import argparse
import os
import re
from typing import NamedTuple, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.rubin_helpers as sm107_utils
import torch
from cutlass._mlir.dialects import math
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05.mma import CollectorOp
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    fmin,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    silu_f32,
)


class S2TCopyBundle(NamedTuple):
    """Bundle of tiled copy and partitioned tensors for smem-to-tmem copies."""

    tiled_copy: cute.TiledCopy
    sSF_compact: cute.Tensor  # Partitioned source (smem)
    tSF_compact: cute.Tensor  # Partitioned destination (tmem)


"""
High-performance persistent blockscaled contiguous grouped dense GEMM with gather and SwiGLU fusion
(C = up * silu(gate), where up and gate come from interleaved weight matrix B)
example for the NVIDIA Rubin (SM107) architecture using CUTE DSL.

This kernel performs FC1 layer computation with SwiGLU activation fusion:
1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
3. Optional Quant: When c_dtype is Float4E2M1FN, generates scale factor C and quantizes output

- Matrix A is MxKx1, A can be row-major("K"), ValidM is composed of valid m in different groups
- Matrix B is NxKxL, B can be column-major("K"), L is grouped dimension (number of experts)
  - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]
- Matrix C is Mx(N/2)x1, C can be row-major("N"), N is halved due to SwiGLU fusion
- Matrix SFA layout is filled internally according to A shape and BlockScaledBasicChunk,
  which has M×ceil_div(K, sf_vec_size)×1 elements
- Matrix SFB layout is filled internally according to B shape and BlockScaledBasicChunk,
  which has N×ceil_div(K, sf_vec_size)×L elements
- Token ID mapping tensor enables gather operation for A and SFA

Matrix A/C Memory Layout Diagrams:

   ```
    Group 0    Group 1   Group 2
   -+---------+---------+---------+
    |         |         |         |
   K| ValidM0 | ValidM1 | ValidM2 |
    |         |         |         |
   -+---------+---------+---------+
    |<-        ValidM           ->|
   ```
   Note: the Group(L) dimension will be flatted into M dimension, and the rest Group(L) size is 1.
         each ValidM will be aligned to 256 or 128. The alignment is determined by the mma_tiler_mn parameter.
         For NVFP4, 2CTA, the alignment is 256. For NVFP4, 1CTA, the alignment is 128.

This GEMM kernel supports the following features:
    - Utilizes LDGSTS (Load Global to Shared with Swizzle) for A and SFA with gather operation
    - Utilizes Tensor Memory Access (TMA) for B and SFB matrices
    - Utilizes Blackwell's tcgen05.mma for matrix multiply-accumulate (MMA) operations
    - Implements TMA multicast with cluster to reduce L2 memory traffic
    - Support persistent tile scheduling to better overlap memory load/store with mma between tiles
    - Support warp specialization to avoid explicit pipelining between mainloop load and mma

This GEMM works as follows:
1. SCHEDULER warp (warp 10): Dispatches tile information to all consumer warps via tile_info_pipeline.
2. LDGSTS A/SFA warps (warps 4-7):
    - Load A matrix from global memory (GMEM) to shared memory (SMEM) using LDGSTS instructions with gather.
    - Load SFA (scale factor A) from GMEM to SMEM using LDGSTS instructions.
    - Uses token_id_mapping to perform permutation/gather during load.
3. TMA B/SFB warp (warp 9):
    - Load B and SFB matrices from GMEM to SMEM using TMA operations with multicast.
4. MMA warp (warp 8):
    - Load scale factor A/B from shared memory (SMEM) to tensor memory (TMEM) using tcgen05.cp instruction.
    - Perform matrix multiply-accumulate (MMA) operations using tcgen05.mma instruction.
5. EPILOGUE warps (warps 0-3):
    - Load two accumulator subtiles (up and gate) from tensor memory (TMEM) to registers (RMEM) using tcgen05.ld.
    - Apply alpha scaling: up_scaled = alpha * up, gate_scaled = alpha * gate
    - Compute SwiGLU activation: output = up_scaled * silu(gate_scaled), where silu(x) = x * sigmoid(x)
    - If c_dtype is Float4E2M1FN: generate scale factor C (SFC) and quantize output
    - Type convert output to c_dtype.
    - Store C matrix from registers (RMEM) to shared memory (SMEM) to global memory (GMEM) with TMA operations.

SM100 tcgen05.mma.kind.block_scale instructions operate as follows:
- Read matrix A from SMEM
- Read matrix B from SMEM
- Read scalefactor A from TMEM
- Read scalefactor B from TMEM
- Write accumulator to TMEM
The accumulator in TMEM must then be loaded to registers before writing back to GMEM.

Constraints:
* Supported input data types: mxf8, mxf4, nvf4
  see detailed valid dtype combinations in below Sm100BlockScaledPersistentDenseGemmKernel class documentation
* A/B tensor must have the same data type, mixed data type is not supported (e.g., mxf8 x mxf4)
* Mma tiler M must be 128 or 256(use_2cta_instrs)
* Mma tiler N must be 64/128/192/256
* Cluster shape M/N must be positive and power of 2, total cluster size <= 16
* Cluster shape M must be multiple of 2 if Mma tiler M is 256(use_2cta_instrs)
* The contiguous dimension of A/B/C tensors must be at least 16 bytes aligned,
  i.e, number of elements is a multiple of 16 and 32 for Float8 and Float4, respectively.

CUDA Graph Support:
* For CUDA graph support, the tile_idx_to_expert_idx, token_id_mapping, A/C matrices,
  and scale factor A can be padded to a larger size
  (e.g., permuted_m = m*topK + num_local_experts*(256-1),
  example: 4096*8 + (256/32)*255 = 34808)
* Use create_tensors() with permuted_m parameter to automatically pad:
  - tile_idx_to_expert_idx: padded for invalid tiles (set to -2e9 for padding tiles)
  - token_id_mapping: padded to permuted_m size (invalid tokens set to -1)
  - A matrix: padded to permuted_m rows (padding rows contain dummy data)
  - C matrix: padded to permuted_m rows (output buffer for cuda_graph)
  - Scale factor A: padded to match A matrix dimensions
* Kernel handling of padding:
  - Scheduler warp checks if tile_idx >= num_non_exiting_tiles to exit
  - Only valid tiles (tile_idx < num_non_exiting_tiles) are written to tile_info pipeline
  - LDGSTS warps use token_id_mapping predicates to skip invalid tokens (token_id == -1)
  - When no more valid tiles exist, outer loop exits and calls producer_tail()
  - Consumer warps process only valid tiles from pipeline
  - No deadlock or synchronization issues
* Consumer warps check initial tile against num_non_exiting_tiles and set
  is_valid_tile=False if tile_idx >= num_non_exiting_tiles
* Only rows within (aligned_groupm[0]+aligned_groupm[1]+...) contain valid data
* Padding rows in C matrix will not be written by the kernel
"""


class Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel:
    """Rubin (SM107) contiguous grouped matrix multiplication with gather operation and SwiGLU fusion
    for FC1 layer computation (C = up * silu(gate), where up/gate come from interleaved GEMM result).

    The computation flow:
    1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
    2. SwiGLU: C = up * silu(gate), extracted from interleaved acc with granularity=64
    3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

    Note: Output C has N/2 columns since pairs of (up, gate) are combined by SwiGLU.

    Key Features:
    - Uses LDGSTS instructions for loading A and SFA matrices with gather/permutation capability
    - Uses TMA (Tensor Memory Access) for loading B and SFB matrices with multicast
    - Token ID mapping enables efficient gather operation during A/SFA load
    - SwiGLU activation fusion in epilogue (up * silu(gate) with interleaved weights)
    - Optional quantization fusion for Float4E2M1FN output with scale factor generation
    - Support for B-reuse pattern (Bkeep-Breuse)
    - Warp specialization: Scheduler (warp 10), A Sync Transform (warp 11, only used when
      use_2cta_instrs is True), LDGSTS A/SFA (warps 4-7), TMA B/SFB (warp 9), MMA (warp 8),
      Epilogue (warps 0-3)

    :param sf_vec_size: Scale factor vector size (16 or 32)
    :param mma_inst_shape: Shape of MMA instruction (M, N, K)
    :param mma_tiler: Shape of MMA tiler (M, N, K)
    :param cluster_shape_mn: Cluster dimensions (M, N)
    :param vectorized_f32: Whether to use vectorized f32x2 operations
    :param topk: Number of experts selected per token
    :param raster_along_m: If True, raster tiles along M dimension first
    """

    def __init__(
        self,
        sf_vec_size: int,
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        vectorized_f32: bool,
        topk: cutlass.Int64,
        raster_along_m: bool = False,
        enable_pdl: bool = True,
    ):
        self.sf_vec_size = sf_vec_size
        self.enable_pdl = enable_pdl
        self.topk = topk
        self.acc_dtype = cutlass.Float32
        self.mma_inst_shape = mma_inst_shape
        self.mma_tiler = mma_tiler
        self.cluster_shape_mn = cluster_shape_mn
        self.raster_along_m = raster_along_m

        self.use_2cta_instrs = mma_inst_shape[0] == 256
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.arch = "sm_107"
        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.ldgsts_a_warp_id = (
            4,
            5,
            6,
            7,
        )
        self.mma_warp_id = 8
        self.tma_b_warp_id = 9
        self.sched_warp_id = 10
        self.sync_transform_warp_id = 11
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * len(
            (
                self.mma_warp_id,
                *self.ldgsts_a_warp_id,
                self.tma_b_warp_id,
                *self.epilog_warp_id,
                self.sched_warp_id,
                self.sync_transform_warp_id,
            )
        )
        self.warps_wo_sched = (
            len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    self.sync_transform_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
            if self.use_2cta_instrs
            else len(
                (
                    *self.epilog_warp_id,
                    self.mma_warp_id,
                    self.tma_b_warp_id,
                    *self.ldgsts_a_warp_id,
                )
            )
        )
        self.threads_wo_sched = self.threads_per_warp * self.warps_wo_sched

        # Set barrier for cta sync, epilogue sync and tmem ptr sync
        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_cta,
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=32 * len(self.epilog_warp_id),
        )
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=32 * len((self.mma_warp_id, *self.epilog_warp_id)),
        )
        self.sched_sync_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp,
        )

        self.num_smem_capacity = self.smem_capacity
        # num_tmem_alloc_cols already set in __init__

        self.vectorized_f32 = vectorized_f32

        # For epilogue compatibility
        self.epilogue_warp_id = self.epilog_warp_id

        # B-reuse pattern control
        self.enable_breuse = mma_tiler[0] // mma_inst_shape[0] == 2

    def _setup_attributes(self):
        """Set up configurations that are dependent on GEMM inputs

        This method configures various attributes based on the input tensor properties
        (data types, leading dimensions) and kernel settings:
        - Configuring tiled MMA
        - Computing MMA/cluster/tile shapes
        - Computing cluster layout
        - Computing multicast CTAs for A/B
        - Computing epilogue subtile
        - Setting up A/B/C stage counts in shared memory
        - Computing A/B/C shared memory layout
        - Computing tensor memory allocation columns
        """

        self.mma_inst_shape_sfb = (
            self.mma_inst_shape[0] // (2 if self.use_2cta_instrs else 1),
            cute.round_up(self.mma_inst_shape[1], 128),
            self.mma_inst_shape[2],
        )

        # Configure tiled mma (Rubin SM107)
        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=(1, 1, 1),
            permutation_mnk=self._get_mma_permutation_mnk(),
        )

        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )

        # Compute mma/cluster/tile shapes
        self.mma_tiler_sfb = (
            self.mma_inst_shape_sfb[0],
            self.mma_inst_shape_sfb[1],
            self.mma_tiler[2],
        )

        self.mma_tiler_c = (
            self.mma_tiler[0],
            self.mma_tiler[1] // 2,
            self.mma_tiler[2],
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        # Number of LDGSTS.128 loads per thread for A matrix (each loads 16 M-rows)
        self.a_num_loads = self.cta_tile_shape_mnk[0] // 16

        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cta_tile_shape_mnk_c = (
            self.mma_tiler_c[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_c[1],
            self.mma_tiler_c[2],
        )

        # Compute SFA tiler for LDGSTS gather (use mma_inst_shape for M/N, scaled K for SF)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.mma_tiler[2] // mma_inst_shape_k
        self.mma_tiler_sfa = (
            self.mma_inst_shape[0],
            self.mma_inst_shape[1],
            mma_inst_shape_k * mma_inst_tile_k // 16,
        )
        self.cta_tile_shape_mnk_sfa = (
            self.mma_tiler_sfa[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfa[1],
            self.mma_tiler_sfa[2],
        )

        # Compute cluster layout
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Compute number of multicast CTAs for A/B
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Fixed epilogue tile for SwiGLU: (128, 64)
        # SwiGLU halves output N, so the default SM107_TILES lookup (keyed on full
        # cta_n) can produce epi_tile_n that is too small (e.g. 32 for 2CTA+N=256),
        # causing wrong TMA store strides and insufficient SFC elements for
        # cvt_fptrunc 32-bit alignment. A fixed (128, 64) works for all configs.
        self.epi_tile = (128, 64)
        self.epi_tile_n = cute.size(self.epi_tile[1])
        self.epi_tile_cnt = (
            self.cta_tile_shape_mnk_c[0] // cute.size(self.epi_tile[0]),
            self.cta_tile_shape_mnk_c[1] // cute.size(self.epi_tile[1]),
        )

        # Setup A/B/C/Scale stage count in shared memory and ACC stage count in tensor memory
        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_c_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.sf_dtype,
            self.sf_vec_size,
            self.smem_capacity,
            self.occupancy,
            self.enable_breuse,
        )

        # Compute A/B/C/Scale shared memory layout
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )

        # Precompute SFA SMEM phase stride for B-reuse LDGSTS addressing
        # This is the element offset between MMA_M=0 and MMA_M=1 in the SFA SMEM layout
        self.sfa_smem_mma_m_stride = (
            int(self.sfa_smem_layout_staged.stride[1]) if self.enable_breuse else 0
        )

        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )

        # Compute TMEM layouts for SFA/SFB (Rubin precomputed)
        self.tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0)),
        )
        self.tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(self.sfb_smem_layout_staged, (None, None, None, 0)),
        )

        # Compute TMEM column counts
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

    def _get_mma_permutation_mnk(self):
        if cutlass.const_expr(self.use_2cta_instrs and self.enable_breuse):
            m_layout = cute.make_layout(
                shape=(self.mma_inst_shape[0] // 2, 2, 2),
                stride=(1, self.mma_inst_shape[0], self.mma_inst_shape[0] // 2),
            )
            return (m_layout, self.mma_inst_shape[1], self.mma_inst_shape[2])
        else:
            return (1, 1, 1)

    def _is_interleaved_utccp(self) -> bool:
        """Enable interleaving UTCCP for Bkeep-Breuse case for 4xFP4 kernel."""
        return (
            self.a_dtype.width == 4 and self.b_dtype.width == 4 and self.enable_breuse
        )

    def _mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> S2TCopyBundle:
        """Make tiledCopy for smem to tmem load for scale factor tensor."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

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
        self,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Copy SFA/SFB from smem to tmem."""
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
        self,
        k_block: int,
        stage_idx: int,
        sfa_s2t_bundle: S2TCopyBundle,
        sfb_s2t_bundle: S2TCopyBundle,
    ):
        """Interleaved UTCCP for Bkeep-Breuse pattern."""
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
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        sfc_tensor: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Execute the contiguous grouped GEMM with gather operation and SwiGLU fusion.

        This method performs FC1 layer computation:
        1. GEMM: acc = alpha * (SFA * A[token_ids]) * (SFB * B)
        2. SwiGLU: C = up * silu(gate), where up/gate are extracted from interleaved acc (granularity=64)
        3. Optional Quant: When c_dtype is Float4E2M1FN, generates SFC and quantizes output

        Data loading:
        - A and SFA are loaded using LDGSTS instructions with token-based gather
        - B and SFB are loaded using TMA instructions with multicast
        - B weights are interleaved: [up_0:64, gate_64:128, up_128:192, gate_192:256, ...]

        Execution steps:
        1. Setup static attributes before smem/grid computation
        2. Setup TMA load/store atoms for B, SFB, and C (no TMA for A/SFA)
        3. Compute grid size with regard to hardware constraints
        4. Define shared storage for kernel
        5. Launch the kernel synchronously with warp specialization:
           - Scheduler warp: Dispatches tile information
           - LDGSTS warps: Load A and SFA with gather
           - A Sync Transform warps: Transform the sync signal of A and SFA from global to
             shared memory when use_2cta_instrs is True
           - TMA warp: Load B and SFB with multicast
           - MMA warp: Perform matrix multiply-accumulate
           - Epilogue warps: Apply SwiGLU activation, optional quantization, and store results

        :param a: Input tensor A (MxKx1), will be gathered using token_id_mapping
        :type a: cute.Tensor
        :param b: Input tensor B (NxKxL), L is the number of experts/groups, weights are interleaved for SwiGLU
        :type b: cute.Tensor
        :param c: Output tensor C (Mx(N/2)x1), N is halved due to SwiGLU fusion
        :type c: cute.Tensor
        :param sfa: Scale factor tensor A, will be gathered using token_id_mapping
        :type sfa: cute.Tensor
        :param sfb: Scale factor tensor B
        :type sfb: cute.Tensor
        :param sfc_tensor: Scale factor tensor C for quantized output (None if not quantizing)
        :type sfc_tensor: Optional[cute.Tensor]
        :param norm_const_tensor: Normalization constant for scale factor generation
            (None if not quantizing)
        :type norm_const_tensor: Optional[cute.Tensor]
        :param tile_idx_to_expert_idx: Mapping from tile index to expert ID,
            shape (permuted_m/cta_tile_m,) where cta_tile_m is the CTA tile M size
        :type tile_idx_to_expert_idx: cute.Tensor
        :param tile_idx_to_mn_limit: Mapping from tile index to M-N dimension limit
            for boundary checking, shape (permuted_m/cta_tile_m,)
        :type tile_idx_to_mn_limit: cute.Tensor
        :param token_id_mapping_tensor: Token ID mapping for gather operation, shape (permuted_m,)
        :type token_id_mapping_tensor: cute.Tensor
        :param num_non_exiting_tiles: Number of valid tiles to process (valid_m/cta_tile_m), shape (1,)
        :type num_non_exiting_tiles: cute.Tensor
        :param alpha: Alpha tensor for each group
        :type alpha: cute.Tensor
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: cutlass.Constexpr
        :param stream: CUDA stream for asynchronous execution
        :type stream: cuda.CUstream
        :param epilogue_op: Optional elementwise lambda function to apply to the output tensor
        :type epilogue_op: cutlass.Constexpr
        :raises TypeError: If input data types are incompatible with the MMA instruction.
        """
        # Setup static attributes before smem/grid/tma computation
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.c_dtype: Type[cutlass.Numeric] = c.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        # Note: Rubin supports mixed A/B dtypes (e.g., Float8E4M3FN x Float8E5M2)

        # Setup attributes that dependent on gemm inputs
        self._setup_attributes()

        # Setup sfb tensor by filling B tensor to scale factor atom layout
        # ((Atom_N, Rest_N),(Atom_K, Rest_K),RestL)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b.shape, self.sf_vec_size)
        sfb = cute.make_tensor(sfb.iterator, sfb_layout)

        # Setup sfc tensor by filling C tensor to scale factor atom layout
        self.generate_sfc = sfc_tensor is not None and norm_const_tensor is not None
        if cutlass.const_expr(self.generate_sfc):
            sfc_layout = blockscaled_utils.tile_atom_to_shape_SF(
                c.shape, self.sf_vec_size
            )
            sfc_tensor = cute.make_tensor(sfc_tensor.iterator, sfc_layout)

        atom_layout_mnk = (1, 1, 1)
        permutation_mnk = self._get_mma_permutation_mnk()

        tiled_mma = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
            atom_layout_mnk=atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )
        tiled_mma.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma.set(tcgen05.Field.NEGATE_B, False)

        # For 2CTA blockscaled kernels, SFB needs to be replicated across peer CTAs.
        tiled_mma_sfb = sm107_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cute.nvgpu.tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_sfb,
            a_collector_op=CollectorOp.DISCARD,
            b_collector_op=CollectorOp.DISCARD,
        )
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_A, False)
        tiled_mma_sfb.set(tcgen05.Field.NEGATE_B, False)

        tiled_mma_bkeep = None
        tiled_mma_breuse = None
        if cutlass.const_expr(self.enable_breuse):
            tiled_mma_bkeep = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.FILL,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_bkeep.set(tcgen05.Field.NEGATE_B, False)

            tiled_mma_breuse = sm107_utils.make_blockscaled_trivial_tiled_mma(
                self.a_dtype,
                self.b_dtype,
                self.a_major_mode,
                self.b_major_mode,
                self.sf_dtype,
                self.sf_vec_size,
                self.cta_group,
                self.mma_inst_shape,
                a_collector_op=CollectorOp.DISCARD,
                b_collector_op=CollectorOp.LASTUSE,
                atom_layout_mnk=atom_layout_mnk,
                permutation_mnk=permutation_mnk,
            )
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_A, False)
            tiled_mma_breuse.set(tcgen05.Field.NEGATE_B, False)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Setup TMA load for B
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # Setup TMA load for SFB
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            sfb,
            sfb_smem_layout,
            self.mma_tiler_sfb,
            tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        # This modifies the layout to handle overlapping 256x(# of scale factors for a single column of B (nNSF))
        # logical blocks for SFB when cta_tile_shape_n=192.
        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb.shape[0][1], 4)

            new_shape = (
                (tma_tensor_sfb.shape[0][0], ((2, 2), y)),
                tma_tensor_sfb.shape[1],
                tma_tensor_sfb.shape[2],
            )
            # Use right multiplication for ScaledBasis (3 * x instead of x * 3)
            x_times_3 = 3 * x
            new_stride = (
                (tma_tensor_sfb.stride[0][0], ((x, x), x_times_3)),
                tma_tensor_sfb.stride[1],
                tma_tensor_sfb.stride[2],
            )
            tma_tensor_sfb_new_layout = cute.make_layout(new_shape, stride=new_stride)
            tma_tensor_sfb = cute.make_tensor(
                tma_tensor_sfb.iterator, tma_tensor_sfb_new_layout
            )

        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size

        # Setup TMA store for C
        tma_atom_c = None
        tma_tensor_c = None
        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c,
            epi_smem_layout,
            self.epi_tile,
        )

        # Compute grid size
        self.tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_mnk_c,
            self.cluster_shape_mn,
            max_active_clusters,
            self.raster_along_m,
        )

        self.buffer_align_bytes = 1024

        # Define shared storage for kernel
        @cute.struct
        class SharedStorage1cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        @cute.struct
        class SharedStorage2cta:
            # (bidx, bidy, bidz, valid, mn_limit)
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage],
                # 1 byte alignment
                1,
            ]
            a_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            a_sync_transform_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # (EPI_TILE_M, EPI_TILE_N, STAGE)
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype,
                    cute.cosize(self.c_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_M, MMA_K, STAGE)
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (MMA, MMA_N, MMA_K, STAGE)
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged.outer)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = (
            SharedStorage2cta
            if cutlass.const_expr(self.use_2cta_instrs)
            else SharedStorage1cta
        )

        # Launch the kernel synchronously
        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            tiled_mma_sfb,
            a,
            tma_atom_b,
            tma_tensor_b,
            sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            tma_atom_c,
            tma_tensor_c,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping_tensor,
            num_non_exiting_tiles,
            alpha,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.tCtSFA_layout,
            self.tCtSFB_layout,
            self.c_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
            epilogue_op,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[union-attr]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for smem to tmem load for scale factor tensor, then use it to
        partition smem memory (source) and tensor memory (destination).

        :param sSF: The scale factor tensor in smem
        :type sSF: cute.Tensor
        :param tSF: The scale factor tensor in tmem
        :type tSF: cute.Tensor

        :return: A tuple containing (tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t) where:
            - tiled_copy_s2t: The tiled copy operation for smem to tmem load for scale factor tensor(s2t)
            - tCsSF_compact_s2t: The partitioned scale factor tensor in smem
            - tSF_compact_s2t: The partitioned scale factor tensor in tmem
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        # (MMA, MMA_MN, MMA_K, STAGE)
        tCsSF_compact = cute.filter_zeros(sSF)
        # (MMA, MMA_MN, MMA_K)
        tCtSF_compact = cute.filter_zeros(tSF)

        # Make S2T CopyAtom and tiledCopy
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K, STAGE)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        # ((ATOM_V, REST_V), Rest_Tiler, MMA_MN, MMA_K)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # GPU device kernel
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: Optional[cute.TiledMma],
        tiled_mma_breuse: Optional[cute.TiledMma],
        tiled_mma_sfb: cute.TiledMma,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mSFC_mnl: Optional[cute.Tensor],
        norm_const_tensor: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping_tensor: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        alpha: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        tCtSFA_layout: cute.Layout,
        tCtSFB_layout: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        epilogue_op: cutlass.Constexpr,
    ):
        """
        GPU device kernel performing the Persistent batched GEMM computation.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        #
        # Prefetch tma desc
        #
        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        #
        # Setup cta/thread coordinates
        #
        # Coords inside cluster
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # Coord inside cta
        tidx, _, _ = cute.arch.thread_idx()

        #
        # Alloc and init: a+b full/empty, accumulator full/empty, tensor memory dealloc barrier
        #
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # Pipeline Init: Initialize A pipeline for LDGSTS operations
        # Producer: 4 warps (warps 4-7) with 128 threads total for LDGSTS operations
        # Consumer: MMA warp for consuming A/SFA data
        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 4,
        )

        a_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.a_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize A SYNC Transform pipeline when use_2cta_instrs is True
        # Producer: 1 warp (warp 11) for LDGSTS SYNC transformation operations
        # Consumer: MMA warp for consuming A/SFA data
        if cutlass.const_expr(self.use_2cta_instrs):
            a_sync_transform_pipeline_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * cute.size(cluster_layout_vmnk, mode=[0]),
            )
            a_sync_transform_pipeline = pipeline.PipelineAsyncUmma.create(
                barrier_storage=storage.a_sync_transform_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=a_sync_transform_pipeline_producer_group,
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )

        # Pipeline Init: Initialize B pipeline for TMA operations
        # Using PipelineTmaUmma for B/SFB since they use TMA load with multicast support
        # Producer: TMA B/SFB warp (warp 9) - 1 warp issuing TMA operations
        # Consumer: MMA warp for consuming B/SFB data
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_b
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        b_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,  # Total bytes loaded by TMA (B + SFB)
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init: Initialize acc_pipeline (barrier) and states
        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (
            2 if use_2cta_instrs else 1
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # Pipeline Init:Initialize tile info pipeline (barrier) and states
        tile_info_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_per_warp * 1,
        )
        tile_info_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.threads_wo_sched,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=tile_info_pipeline_producer_group,
            consumer_group=tile_info_pipeline_consumer_group,
        )

        # Tensor memory dealloc barrier init
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
            arch=self.arch,
        )

        # Cluster arrive after barrier init (Rubin uses pipeline_init_arrive)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        #
        # Setup smem tensor A/B/C/Scale
        #
        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sC = storage.sC.get_tensor(
            c_smem_layout_staged.outer, swizzle=c_smem_layout_staged.inner
        )
        # (MMA, MMA_M, MMA_K, STAGE)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        # (MMA, MMA_N, MMA_K, STAGE)
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # (granularity_m, repeat_m), (granularity_k, repeat_k), num_scale_stage)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        # (granularity_n, repeat_n), (granularity_k, repeat_k), num_scale_stage)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        # (bidx, bidy, bidz, valid, mn_limit)
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        #
        # Compute multicast mask for A/B buffer full
        #
        b_full_mcast_mask = None
        sfb_full_mcast_mask = None
        if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )
            sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1
            )

        #
        # Local_tile partition global tensors
        #
        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        # (bN, bK, loopN, loopK, loopL)
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )

        # (bM, bK, RestM, RestK, RestL)
        gSFA_mkl = cute.local_tile(
            mSFA_mkl,
            cute.slice_(self.cta_tile_shape_mnk_sfa, (None, 0, None)),
            (None, None, None),
        )

        # (bN, bK, RestN, RestK, RestL)
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
            (None, None, None),
        )

        gToken_ml = cute.local_tile(
            token_id_mapping_tensor,
            cute.slice_(self.cta_tile_shape_mnk, (None, 0, 0)),
            (None,),
        )

        # (bM, bN, loopM, loopN, loopL)
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler_c, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        #
        # Partition global tensor for TiledMMA_A/B/C
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        # (MMA, MMA_N, MMA_K, loopN, loopK, loopL)
        tCgB = thr_mma.partition_B(gB_nkl)
        # (MMA, MMA_N, MMA_K, RestN, RestK, RestL)
        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
        # (MMA, MMA_M, MMA_N, loopM, loopN, loopL)
        tCgC = thr_mma.partition_C(gC_mnl)

        #
        # Partition global/shared tensor for TMA load B
        #
        # TMA load B partition_S/D
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), loopM, loopK, loopL)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        # TMA load SFB partition_S/D
        sfb_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
        )
        # ((atom_v, rest_v), STAGE)
        # ((atom_v, rest_v), RestN, RestK, RestL)
        tBsSFB, tBgSFB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_sfb_vmnk[1],
            sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(tCgSFB, 0, 3),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        #
        # Partition shared/tensor memory tensor for TiledMMA_A/B/C
        #
        # (MMA, MMA_M, MMA_K, STAGE)
        tCrA = tiled_mma.make_fragment_A(sA)
        # (MMA, MMA_N, MMA_K, STAGE)
        tCrB = tiled_mma.make_fragment_B(sB)
        # (MMA, MMA_M, MMA_N)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, STAGE)
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        griddepcontrol_wait()

        #
        # Specialized Schedule Warp
        #
        if warp_idx == self.sched_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            tile_info_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_tile_stage
            )

            num_non_exiting_tiles_value = num_non_exiting_tiles[0]

            if cutlass.const_expr(self.raster_along_m):
                while work_tile.is_valid_tile:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()
            else:
                is_continue = cutlass.Boolean(1)
                while work_tile.is_valid_tile and is_continue:
                    cur_tile_coord = work_tile.tile_idx
                    mma_tile_coord_m = cur_tile_coord[0] // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    if mma_tile_coord_m < num_non_exiting_tiles_value:
                        tile_info_pipeline.producer_acquire(tile_info_producer_state)
                        cur_tile_coord = work_tile.tile_idx
                        expert_idx = tile_idx_to_expert_idx[mma_tile_coord_m]
                        mn_limit = tile_idx_to_mn_limit[mma_tile_coord_m]
                        with cute.arch.elect_one():
                            sInfo[(0, tile_info_producer_state.index)] = cur_tile_coord[
                                0
                            ]
                            sInfo[(1, tile_info_producer_state.index)] = cur_tile_coord[
                                1
                            ]
                            sInfo[(2, tile_info_producer_state.index)] = expert_idx
                            sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(
                                work_tile.is_valid_tile
                            )
                            sInfo[(4, tile_info_producer_state.index)] = mn_limit
                            # fence view async shared
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )

                        self.sched_sync_barrier.arrive_and_wait()
                        tile_info_pipeline.producer_commit(tile_info_producer_state)
                        tile_info_producer_state.advance()
                    else:
                        is_continue = cutlass.Boolean(0)

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = work_tile.tile_idx[0]
                sInfo[(1, tile_info_producer_state.index)] = work_tile.tile_idx[1]
                sInfo[(2, tile_info_producer_state.index)] = -1
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = -1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # Specialized LDGSTS A/SFA warps (warps 4-7)
        # These warps use LDGSTS instructions to load A and SFA from global to shared memory
        # with gather/permutation capability enabled by token_id_mapping
        #
        if (
            warp_idx <= self.ldgsts_a_warp_id[-1]
            and warp_idx >= self.ldgsts_a_warp_id[0]
        ):
            #
            # Setup LDGSTS copy atoms for A and SFA
            # A: 8x LDGSTS.128 per thread with swizzle_128B for A matrix (32 elements per thread)
            # SFA: 4x LDGSTS.32 per thread with 512-element block swizzling for scale factor A (4 elements per thread)
            #
            a_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                mA_mkl.element_type,
                num_bits_per_copy=128,
            )
            a_thread_layout = cute.make_layout((16, 8), stride=(8, 1))
            a_value_layout = cute.make_layout((1, 32), stride=(32, 1))
            a_tiled_copy = cute.make_tiled_copy_tv(
                a_atom_copy,
                a_thread_layout,
                a_value_layout,
            )

            sfa_atom_copy = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mSFA_mkl.element_type,
                num_bits_per_copy=32,
            )
            tidx_in_warpgroup = tidx % 128

            sA_tiled = cute.make_tensor(
                sA.iterator,
                layout=cute.make_layout(
                    (
                        self.cta_tile_shape_mnk[0],
                        self.cta_tile_shape_mnk[2],
                        self.num_ab_stage,
                    ),
                    stride=(
                        self.cta_tile_shape_mnk[2],
                        1,
                        self.cta_tile_shape_mnk[0] * self.cta_tile_shape_mnk[2],
                    ),
                ),
            )
            a_thr_copy = a_tiled_copy.get_slice(tidx_in_warpgroup)
            tAsA_tiled = a_thr_copy.partition_D(sA_tiled)

            a_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.a_num_loads,)),
                cutlass.Int32,
            )
            a_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((self.a_num_loads,)),
                cutlass.Boolean,
            )
            sfa_phase_cnt = 2 if self.enable_breuse else 1
            sfa_token_offset_tensor = cute.make_rmem_tensor(
                cute.make_layout((sfa_phase_cnt,)),
                cutlass.Int32,
            )
            sfa_predicate_tensor = cute.make_rmem_tensor(
                cute.make_layout((sfa_phase_cnt,)),
                cutlass.Boolean,
            )
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(5, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Load token IDs for gather operation
                # For A: each thread loads a_num_loads token offsets
                # For SFA: each thread loads 1 (or 2 for B-reuse) token offsets
                gToken_ml_tile = gToken_ml[(None, tile_info[0])]
                for i in range(self.a_num_loads):
                    token_ml_tile_offset = (tidx_in_warpgroup // 8) + i * 16
                    a_token_offset_tensor[i] = gToken_ml_tile[token_ml_tile_offset]
                    a_predicate_tensor[i] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )
                    a_token_offset_tensor[i] = (
                        a_token_offset_tensor[i] // self.topk
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + token_ml_tile_offset
                        < tile_info[4]
                        else 0
                    )

                token_ml_tile_offset = (
                    8 * (tidx_in_warpgroup // 32)
                    + 32 * ((tidx_in_warpgroup % 32) // 8)
                    + (tidx_in_warpgroup % 8)
                )
                sfa_token_offset_tensor[0] = (
                    gToken_ml_tile[token_ml_tile_offset] // self.topk
                )
                sfa_predicate_tensor[0] = (
                    cutlass.Boolean(1)
                    if tile_info[0] * self.cta_tile_shape_mnk[0] + token_ml_tile_offset
                    < tile_info[4]
                    else cutlass.Boolean(0)
                )
                relative_sfa_token_offset = sfa_token_offset_tensor[0]

                # B-reuse: load SFA phase 1 token (offset by half the CTA M-tile)
                # cta_tile_shape_mnk[0] // 2 is the per-phase M size for both 1CTA and 2CTA
                if cutlass.const_expr(self.enable_breuse):
                    sfa_token_ml_offset_phase1 = (
                        token_ml_tile_offset + self.cta_tile_shape_mnk[0] // 2
                    )
                    sfa_token_offset_tensor[1] = (
                        gToken_ml_tile[sfa_token_ml_offset_phase1] // self.topk
                    )
                    sfa_predicate_tensor[1] = (
                        cutlass.Boolean(1)
                        if tile_info[0] * self.cta_tile_shape_mnk[0]
                        + sfa_token_ml_offset_phase1
                        < tile_info[4]
                        else cutlass.Boolean(0)
                    )

                tAgA = gA_mkl[(None, None, 0, None, 0)]
                A_gmem_thread_offset = cute.assume(
                    (tidx_in_warpgroup % 8) * 32, divby=32
                )
                tAgSFA = gSFA_mkl[(relative_sfa_token_offset, None, 0, None, 0)]

                # Initialize phase 1 SFA GMEM tensor (must have initial value before control flow)
                tAgSFA_phase1 = tAgSFA
                if cutlass.const_expr(self.enable_breuse):
                    tAgSFA_phase1 = gSFA_mkl[
                        (sfa_token_offset_tensor[1], None, 0, None, 0)
                    ]

                tAsSFA = sSFA[
                    (
                        (
                            (
                                (
                                    8 * (tidx_in_warpgroup // 32)
                                    + (tidx_in_warpgroup % 8),
                                    (tidx_in_warpgroup % 32) // 8,
                                ),
                                None,
                            ),
                            None,
                        ),
                        None,
                        None,
                        None,
                    )
                ]

                # Peek (try_wait) SCALE buffer empty
                a_producer_state.reset_count()
                peek_a_empty_status = cutlass.Boolean(1)
                if a_producer_state.count < k_tile_cnt:
                    peek_a_empty_status = a_pipeline.producer_try_acquire(
                        a_producer_state
                    )

                #
                # Load A and SFA with LDGSTS and gather/permutation
                # Each K-tile iteration loads one K-tile of A and SFA from GMEM to SMEM
                # using LDGSTS instructions with token-based gather addressing
                #
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for AB buffer empty
                    a_pipeline.producer_acquire(a_producer_state, peek_a_empty_status)

                    tAgA_ktile = tAgA[(None, None, a_producer_state.count)]
                    tAsA_ktile = tAsA_tiled[(None, None, None, a_producer_state.index)]

                    tAgSFA_ktile = tAgSFA[(None, a_producer_state.count)]
                    tAgSFA_phase1_ktile = tAgSFA_ktile
                    if cutlass.const_expr(self.enable_breuse):
                        tAgSFA_phase1_ktile = tAgSFA_phase1[
                            (None, a_producer_state.count)
                        ]
                    tAsSFA_ktile = tAsSFA[
                        (
                            None,
                            None,
                            None,
                            None,
                            a_producer_state.index,
                        )
                    ]

                    for i in range(self.a_num_loads):
                        #
                        # Load A matrix: a_num_loads x LDGSTS.128 per thread with swizzle_128B
                        # Each LDGSTS.128 loads 32 elements (128 bits) from GMEM to SMEM
                        # Global memory address is computed using token offset for gather operation
                        # Predicate mask guards against invalid token IDs (padding tokens marked as -1)
                        #
                        A_gmem_slice_offset = A_gmem_thread_offset + cute.assume(
                            a_token_offset_tensor[i] * tAgA_ktile.layout[0].stride,
                            divby=32,
                        )
                        A_gmem_slice_offset = cute.assume(A_gmem_slice_offset, divby=32)
                        tAgA_slice_ptr = tAgA_ktile.iterator + A_gmem_slice_offset
                        tAgA_slice = cute.make_tensor(
                            tAgA_slice_ptr, layout=cute.make_layout((32,))
                        )

                        tAsA_slice = cute.make_tensor(
                            tAsA_ktile[(None, i, None)].iterator,
                            layout=cute.make_layout((32,)),
                        )
                        a_predicate_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        a_predicate_slice[0] = a_predicate_tensor[i]

                        cute.copy_atom_call(
                            a_atom_copy, tAgA_slice, tAsA_slice, pred=a_predicate_slice
                        )

                    for phase in range(sfa_phase_cnt):
                        #
                        # Load SFA: 4x LDGSTS.32 per thread per phase with 512-element block swizzling
                        # Each LDGSTS.32 loads 4 scale factor elements (32 bits) from GMEM to SMEM
                        # Uses same token offset as A matrix for consistent gather operation
                        # With B-reuse, we load 2 phases (one per MMA_M half)
                        #
                        tAgSFA_src_ktile = (
                            tAgSFA_ktile if phase == 0 else tAgSFA_phase1_ktile
                        )
                        smem_phase_offset = phase * self.sfa_smem_mma_m_stride
                        sfa_pred_slice = cute.make_rmem_tensor(
                            cute.make_layout((1,)), cutlass.Boolean
                        )
                        sfa_pred_slice[0] = sfa_predicate_tensor[(phase,)]
                        for i in range(4):
                            swizzled_iterator = (tidx_in_warpgroup % 32) // 8 ^ i
                            tAgSFA_slice_ptr = (
                                tAgSFA_src_ktile.iterator + 4 * swizzled_iterator
                            )
                            tAgSFA_slice = cute.make_tensor(
                                tAgSFA_slice_ptr, layout=cute.make_layout((4,))
                            )

                            tAsSFA_slice_ptr = (
                                tAsSFA_ktile.iterator
                                + smem_phase_offset
                                + 512 * swizzled_iterator
                            )
                            tAsSFA_slice = cute.make_tensor(
                                tAsSFA_slice_ptr, cute.make_layout((4,))
                            )

                            cute.copy_atom_call(
                                sfa_atom_copy,
                                tAgSFA_slice,
                                tAsSFA_slice,
                                pred=sfa_pred_slice,
                            )

                    a_pipeline.producer_commit(a_producer_state)

                    # Peek (try_wait) A buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    a_producer_state.advance()
                    peek_a_empty_status = cutlass.Boolean(1)
                    if a_producer_state.count < k_tile_cnt:
                        peek_a_empty_status = a_pipeline.producer_try_acquire(
                            a_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(5, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            #
            # Wait A pipeline buffer empty
            #
            a_pipeline.producer_tail(a_producer_state)

        #
        # Specialized A/SFA Sync Transform Warp (warp 11) when use_2cta_instrs is True
        # This warp serve as sync transformation for A and SFA
        #
        if warp_idx == self.sync_transform_warp_id:
            if cutlass.const_expr(self.use_2cta_instrs):
                #
                # Persistent tile scheduling loop
                #
                tile_sched = utils.StaticPersistentTileScheduler.create(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                # First tile
                work_tile = tile_sched.initial_work_tile_info()

                a_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
                a_sync_transform_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.num_ab_stage
                )
                tile_info_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_tile_stage
                )

                # Get the first tile info
                valid_tile_info = cute.make_rmem_tensor((1,), cutlass.Int32)
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                is_valid_tile = valid_tile_info[0] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

                while is_valid_tile:
                    # Peek (try_wait) A buffer full for k_tile = 0
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )
                    # Peek (try_wait) a sync transform buffer empty
                    a_sync_transform_producer_state.reset_count()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # Conditionally wait for A buffer full
                        a_pipeline.consumer_wait(a_consumer_state, peek_a_full_status)

                        a_sync_transform_pipeline.producer_commit(
                            a_sync_transform_producer_state
                        )
                        a_sync_transform_producer_state.advance()

                        # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    #
                    # Advance to next tile
                    #
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    valid_tile_info[0] = sInfo[(3, tile_info_consumer_state.index)]
                    is_valid_tile = valid_tile_info[0] == 1
                    cute.arch.fence_proxy(
                        "async.shared",
                        space="cta",
                    )
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()

                #
                # Wait A sync transform buffer empty
                #
                a_sync_transform_pipeline.producer_tail(a_sync_transform_producer_state)

        #
        # Specialized TMA B/SFB load warp (warp 9)
        # This warp uses TMA instructions to load B and SFB from global to shared memory
        # with multicast support to reduce L2 memory traffic
        #
        if warp_idx == self.tma_b_warp_id:
            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            # First tile
            work_tile = tile_sched.initial_work_tile_info()

            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Slice to per mma tile index
                #
                # ((atom_v, rest_v), loopK)
                tBgB_slice = tBgB[
                    (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
                ]

                # Apply SFB slicing hack when cta_tile_shape_n=64
                slice_n = mma_tile_coord_mnl[1]
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    slice_n = mma_tile_coord_mnl[1] // 2

                # ((atom_v, rest_v), RestK)
                tBgSFB_slice = tBgSFB[(None, slice_n, None, mma_tile_coord_mnl[2])]

                # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt
                b_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if b_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = b_pipeline.producer_try_acquire(
                        b_producer_state
                    )
                #
                # Tma load loop
                #
                for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    # Conditionally wait for B buffer empty
                    b_pipeline.producer_acquire(b_producer_state, peek_ab_empty_status)

                    tBgB_k = tBgB_slice[(None, b_producer_state.count)]
                    tBgSFB_k = tBgSFB_slice[(None, b_producer_state.count)]
                    tBsB_pipe = tBsB[(None, b_producer_state.index)]
                    tBsSFB_pipe = tBsSFB[(None, b_producer_state.index)]

                    tma_bar = b_pipeline.producer_get_barrier(b_producer_state)

                    # TMA load B
                    cute.copy(
                        tma_atom_b,
                        tBgB_k,
                        tBsB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=b_full_mcast_mask,
                    )

                    # TMA load SFB
                    cute.copy(
                        tma_atom_sfb,
                        tBgSFB_k,
                        tBsSFB_pipe,
                        tma_bar_ptr=tma_bar,
                        mcast_mask=sfb_full_mcast_mask,
                    )

                    # Peek (try_wait) AB buffer empty for k_tile = prefetch_k_tile_cnt + k_tile + 1
                    b_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if b_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = b_pipeline.producer_try_acquire(
                            b_producer_state
                        )

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait A/B buffer empty
            #
            b_pipeline.producer_tail(b_producer_state)

        #
        # Specialized MMA warp
        #
        if warp_idx == self.mma_warp_id:
            #
            # Bar sync for retrieve tensor memory ptr from shared mem
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

            # Make SFA tmem tensor (using precomputed layout)
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # Make SFB tmem tensor (using precomputed layout)
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Partition for S2T copy of SFA/SFB (Rubin uses S2TCopyBundle)
            sfa_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            sfb_s2t_bundle = self._mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            if cutlass.const_expr(self.use_2cta_instrs):
                a_sync_transform_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.num_ab_stage
                )
            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )

            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info from pipeline (scheduler has filtered out tiles >= num_non_exiting_tiles)
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                # Peek (try_wait) AB buffer full for k_tile = 0
                if cutlass.const_expr(self.use_2cta_instrs):
                    a_sync_transform_consumer_state.reset_count()
                    peek_a_sync_transform_full_status = cutlass.Boolean(1)
                    if (
                        a_sync_transform_consumer_state.count < k_tile_cnt
                        and is_leader_cta
                    ):
                        peek_a_sync_transform_full_status = (
                            a_sync_transform_pipeline.consumer_try_wait(
                                a_sync_transform_consumer_state
                            )
                        )
                    a_consumer_state.reset_count()
                else:
                    a_consumer_state.reset_count()
                    peek_a_full_status = cutlass.Boolean(1)
                    if a_consumer_state.count < k_tile_cnt:
                        peek_a_full_status = a_pipeline.consumer_try_wait(
                            a_consumer_state
                        )

                b_consumer_state.reset_count()
                peek_b_full_status = cutlass.Boolean(1)
                if b_consumer_state.count < k_tile_cnt and is_leader_cta:
                    peek_b_full_status = b_pipeline.consumer_try_wait(b_consumer_state)

                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )

                # Get accumulator stage index
                acc_stage_index = acc_producer_state.index

                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                # Apply TMEM pointer offset hack when cta_tile_shape_n=192 or
                # cta_tile_shape_n=64
                tCtSFB_mma = tCtSFB
                if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
                    # If this is an ODD tile, shift the TMEM start address for
                    # cta_tile_shape_n=192 case by two words
                    # (ignores first 64 columns of SFB)
                    offset = (
                        cutlass.Int32(2)
                        if mma_tile_coord_mnl[1] % 2 == 1
                        else cutlass.Int32(0)
                    )
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                elif cutlass.const_expr(self.cta_tile_shape_mnk[1] == 64):
                    # Move in increments of 64 columns of SFB
                    offset = cutlass.Int32((mma_tile_coord_mnl[1] % 2) * 2)
                    shifted_ptr = cute.recast_ptr(
                        acc_tmem_ptr
                        + self.num_accumulator_tmem_cols
                        + self.num_sfa_tmem_cols
                        + offset,
                        dtype=self.sf_dtype,
                    )
                    tCtSFB_mma = cute.make_tensor(shifted_ptr, tCtSFB_layout)
                    #
                # Wait for accumulator buffer empty
                #
                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)
                #
                # Mma mainloop
                #

                for k_tile in cutlass.range(k_tile_cnt):
                    # Set tensor memory buffer for current tile
                    # (MMA, MMA_M, MMA_N)

                    if is_leader_cta:
                        # Conditionally wait for AB buffer full
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_wait(
                                a_sync_transform_consumer_state,
                                peek_a_sync_transform_full_status,
                            )
                        else:
                            a_pipeline.consumer_wait(
                                a_consumer_state, peek_a_full_status
                            )
                        b_pipeline.consumer_wait(b_consumer_state, peek_b_full_status)

                        #  Copy SFA/SFB from smem to tmem and execute MMA (Rubin)
                        if cutlass.const_expr(not self._is_interleaved_utccp()):
                            self._mainloop_s2t_copies(
                                b_consumer_state.index, sfa_s2t_bundle, sfb_s2t_bundle
                            )

                        num_kblocks = cute.size(tCrA, mode=[2])

                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            if cutlass.const_expr(
                                self.enable_breuse
                                and cute.size(tCtAcc.layout, mode=[1]) == 2
                                and cute.size(tCtAcc.layout, mode=[2]) == 1
                            ):
                                tCtAcc_bkeep = tCtAcc[(None, 0, 0)]
                                tCtAcc_breuse = tCtAcc[(None, 1, 0)]

                                a_kblk_crd_keep = (
                                    None,
                                    0,
                                    kblock_idx,
                                    b_consumer_state.index,
                                )
                                a_kblk_crd_reuse = (
                                    None,
                                    1,
                                    kblock_idx,
                                    b_consumer_state.index,
                                )
                                b_kblk_crd = (
                                    None,
                                    0,
                                    kblock_idx,
                                    b_consumer_state.index,
                                )

                                sfa_kblk_crd_keep = (None, 0, kblock_idx)
                                sfa_kblk_crd_reuse = (None, 1, kblock_idx)
                                sfb_kblk_crd = (None, 0, kblock_idx)

                                if cutlass.const_expr(self._is_interleaved_utccp()):
                                    self._mainloop_s2t_interleaved_copies(
                                        kblock_idx,
                                        b_consumer_state.index,
                                        sfa_s2t_bundle,
                                        sfb_s2t_bundle,
                                    )

                                # Bkeep
                                tiled_mma_bkeep.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_bkeep,
                                    tCtAcc_bkeep,
                                    [tCrA[a_kblk_crd_keep], tCtSFA[sfa_kblk_crd_keep]],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_bkeep,
                                )
                                # Breuse
                                tiled_mma_breuse.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma_breuse,
                                    tCtAcc_breuse,
                                    [
                                        tCrA[a_kblk_crd_reuse],
                                        tCtSFA[sfa_kblk_crd_reuse],
                                    ],
                                    [tCrB[b_kblk_crd], tCtSFB_mma[sfb_kblk_crd]],
                                    tCtAcc_breuse,
                                )
                            else:
                                kblock_coord = (
                                    None,
                                    None,
                                    kblock_idx,
                                    b_consumer_state.index,
                                )
                                sf_kblock_coord = (None, None, kblock_idx)

                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    k_tile != 0 or kblock_idx != 0,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    [tCrA[kblock_coord], tCtSFA[sf_kblock_coord]],
                                    [tCrB[kblock_coord], tCtSFB_mma[sf_kblock_coord]],
                                    tCtAcc,
                                )

                        # Async arrive AB buffer empty
                        a_pipeline.consumer_release(a_consumer_state)
                        if cutlass.const_expr(self.use_2cta_instrs):
                            a_sync_transform_pipeline.consumer_release(
                                a_sync_transform_consumer_state
                            )
                        b_pipeline.consumer_release(b_consumer_state)

                    # Peek (try_wait) AB buffer full for k_tile = k_tile + 1
                    if cutlass.const_expr(self.use_2cta_instrs):
                        a_sync_transform_consumer_state.advance()
                        peek_a_sync_transform_full_status = cutlass.Boolean(1)
                        if a_sync_transform_consumer_state.count < k_tile_cnt:
                            if is_leader_cta:
                                peek_a_sync_transform_full_status = (
                                    a_sync_transform_pipeline.consumer_try_wait(
                                        a_sync_transform_consumer_state
                                    )
                                )
                        a_consumer_state.advance()
                    else:
                        a_consumer_state.advance()
                        peek_a_full_status = cutlass.Boolean(1)
                        if a_consumer_state.count < k_tile_cnt:
                            peek_a_full_status = a_pipeline.consumer_try_wait(
                                a_consumer_state
                            )

                    b_consumer_state.advance()
                    peek_b_full_status = cutlass.Boolean(1)
                    if b_consumer_state.count < k_tile_cnt:
                        if is_leader_cta:
                            peek_b_full_status = b_pipeline.consumer_try_wait(
                                b_consumer_state
                            )

                #
                # Async arrive accumulator buffer full(each kblock)
                #
                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)

                # Peek (try_wait) Acc buffer empty for k_tile = k_tile + 1
                acc_producer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Wait for accumulator buffer empty
            #
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Specialized epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            #
            # Alloc tensor memory buffer
            #
            tmem.allocate(self.num_tmem_alloc_cols)

            #
            # Bar sync for retrieve tensor memory ptr from shared memory
            #
            tmem.wait_for_alloc()

            #
            # Retrieving tensor memory ptr and make accumulator tensor
            #
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # (MMA, MMA_M, MMA_N, STAGE)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            #
            # Partition for epilogue (Rubin: transform both accumulator and C layout)
            # transform_partitioned_tensor_layout merges (MMA_ATOM, MMA_M) into flat M.
            # This is unconditional to match the reference kernel pattern.
            #
            tCtAcc_transformed = transform_partitioned_tensor_layout(tCtAcc_base)
            tCgC_for_epi = transform_partitioned_tensor_layout(tCgC)

            epi_tidx = tidx % 128
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc_up,
                tTR_rAcc_gate,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx, tCtAcc_transformed, tCgC_for_epi, epi_tile, use_2cta_instrs
            )

            tTR_rC = None
            tiled_copy_r2s = None
            tRS_rC = None
            tRS_sC = None
            bSG_sC = None
            bSG_gC_partitioned = None
            tTR_rC = cute.make_rmem_tensor(tTR_rAcc_up.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
                self, tiled_copy_t2r, tTR_rC, epi_tidx, sC
            )
            (
                tma_atom_c,
                bSG_sC,
                bSG_gC_partitioned,
            ) = self.epilog_gmem_copy_and_partition(
                epi_tidx, tma_atom_c, tCgC_for_epi, epi_tile, sC
            )

            if cutlass.const_expr(self.generate_sfc):
                norm_const = norm_const_tensor[0]
                # (EPI_TILE_M, EPI_TILE_N, RestM, RestN, RestL)
                gSFC_mnl = cute.local_tile(mSFC_mnl, epi_tile, (None, None, None))

                thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
                # (T2R, T2R_M, T2R_N, RestM, RestN, RestL)
                tCgSFC_mnl = thr_copy_t2r.partition_D(gSFC_mnl)
                tCgSFC_mnl = cute.filter_zeros(tCgSFC_mnl)
                # (T2R, T2R_M, T2R_N)
                tCrSFC = cute.make_rmem_tensor(
                    tCgSFC_mnl[(None, None, None, 0, 0, 0)].layout, self.sf_dtype
                )
                tCrSFC_pvscale = cute.make_rmem_tensor_like(tCrSFC, cutlass.Float32)

            #
            # Persistent tile scheduling loop
            #
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            c_pipeline = None
            # Threads/warps participating in tma store pipeline
            c_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * len(self.epilog_warp_id),
            )
            c_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_c_stage,
                producer_group=c_producer_group,
            )

            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )

            # Get the first tile info
            tile_info = cute.make_rmem_tensor((4,), cutlass.Int32)

            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for idx in cutlass.range(4, unroll_full=True):
                tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy(
                "async.shared",
                space="cta",
            )
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            num_prev_subtiles = cutlass.Int32(0)
            while is_valid_tile:
                mma_tile_coord_mnl = (
                    tile_info[0] // cute.size(tiled_mma.thr_id.shape),
                    tile_info[1],
                    tile_info[2],
                )
                #
                # Get alpha for current group
                #

                expert_idx = mma_tile_coord_mnl[2]
                alpha_val = alpha[expert_idx]

                #
                # Slice to per mma tile index
                #
                bSG_gC = None
                # ((ATOM_V, REST_V), EPI_M, EPI_N)
                bSG_gC = bSG_gC_partitioned[
                    (
                        None,
                        None,
                        None,
                        mma_tile_coord_mnl[0],
                        mma_tile_coord_mnl[1],
                        0,
                    )
                ]

                # Get accumulator stage index
                acc_stage_index = acc_consumer_state.index

                # Set tensor memory buffer for current tile
                # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]

                if cutlass.const_expr(self.generate_sfc):
                    # (T2R, T2R_M, T2R_N, RestM, RestN)
                    tCgSFC_mn = tCgSFC_mnl[
                        (
                            None,
                            None,
                            None,
                            None,
                            None,
                            0,
                        )
                    ]

                #
                # Wait for accumulator buffer full
                #
                acc_pipeline.consumer_wait(acc_consumer_state)

                #
                # Process accumulator subtiles with SwiGLU fusion and store to global memory
                # Each iteration processes a pair of subtiles (up, gate) and computes
                # up * silu(gate)
                #
                # The accumulator has full N columns with interleaved [up, gate] at
                # granularity=64. Output C has N/2 columns. With epi_tile_n, we iterate
                # over M and N output subtiles separately to correctly map up/gate pairs.
                #
                # tTR_tAcc shape: (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE) before group
                # After selecting acc_stage, shape is (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
                # bSG_gC shape: ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
                #   -> after slicing: ((ATOM_V, REST_V), EPI_M, EPI_N)
                #
                interleave_granularity = 64
                gate_offset = interleave_granularity // self.epi_tile_n
                epi_m_cnt = cute.size(tTR_tAcc.shape, mode=[3])
                acc_n_subtile_cnt = cute.size(tTR_tAcc.shape, mode=[4])
                out_n_subtile_cnt = (
                    acc_n_subtile_cnt // 2
                )  # N/2 output subtiles per M subtile

                for epi_m_idx in cutlass.range(epi_m_cnt):
                    for out_n_idx in cutlass.range(out_n_subtile_cnt):
                        # Map output N subtile to accumulator N subtile:
                        # For each interleave block of 2*gate_offset N-subtiles in acc,
                        # first gate_offset subtiles are up, next gate_offset are gate
                        block_idx = out_n_idx // gate_offset
                        within_block = out_n_idx % gate_offset
                        up_n_subtile = block_idx * 2 * gate_offset + within_block
                        gate_n_subtile = (
                            block_idx * 2 * gate_offset + gate_offset + within_block
                        )
                        #
                        # Load accumulator from tensor memory buffer to register
                        #
                        tTR_tAcc_mn_up = tTR_tAcc[
                            (None, None, None, epi_m_idx, up_n_subtile)
                        ]
                        tTR_tAcc_mn_gate = tTR_tAcc[
                            (None, None, None, epi_m_idx, gate_n_subtile)
                        ]

                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_up, tTR_rAcc_up)
                        cute.copy(tiled_copy_t2r, tTR_tAcc_mn_gate, tTR_rAcc_gate)

                        acc_vec_up = tTR_rAcc_up.load()
                        acc_vec_gate = tTR_rAcc_gate.load()

                        #
                        # SwiGLU activation: output = up * silu(gate)
                        # where silu(x) = x * sigmoid(x)
                        # up and gate are extracted from interleaved accumulator subtiles
                        #
                        tCompute = cute.make_rmem_tensor(
                            acc_vec_gate.shape, self.acc_dtype
                        )
                        if cutlass.const_expr(self.vectorized_f32):
                            # SwiGLU Packed Version: uses f32x2 packed operations for better performance
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            # where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
                            LOG2_E = cutlass.Float32(1.4426950408889634)
                            for i in cutlass.range_constexpr(
                                0, cute.size(tTR_rAcc_up), 2
                            ):
                                acc_vec_up_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_up[i], acc_vec_up[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                )
                                acc_vec_gate_alpha = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate[i], acc_vec_gate[i + 1]),
                                    (
                                        cutlass.Float32(alpha_val),
                                        cutlass.Float32(alpha_val),
                                    ),
                                )
                                tCompute_log2e = cute.arch.mul_packed_f32x2(
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                    (-LOG2_E, -LOG2_E),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.add_packed_f32x2(
                                    (
                                        cute.math.exp2(
                                            tCompute_log2e[0], fastmath=True
                                        ),
                                        cute.math.exp2(
                                            tCompute_log2e[1], fastmath=True
                                        ),
                                    ),
                                    (1.0, 1.0),
                                )
                                tCompute[i] = cute.arch.rcp_approx(tCompute[i])
                                tCompute[i + 1] = cute.arch.rcp_approx(tCompute[i + 1])
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_gate_alpha[0], acc_vec_gate_alpha[1]),
                                )
                                (
                                    tCompute[i],
                                    tCompute[i + 1],
                                ) = cute.arch.mul_packed_f32x2(
                                    (tCompute[i], tCompute[i + 1]),
                                    (acc_vec_up_alpha[0], acc_vec_up_alpha[1]),
                                )
                        else:
                            # SwiGLU Unpacked Version: scalar operations
                            # Computes: output = (alpha * up) * silu(alpha * gate)
                            for i in cutlass.range_constexpr(cute.size(tTR_rAcc_up)):
                                acc_vec_up_alpha = acc_vec_up[i] * cutlass.Float32(
                                    alpha_val
                                )
                                acc_vec_gate_alpha = acc_vec_gate[i] * cutlass.Float32(
                                    alpha_val
                                )
                                tCompute[i] = acc_vec_up_alpha * silu_f32(
                                    acc_vec_gate_alpha, fastmath=True
                                )

                        if cutlass.const_expr(self.generate_sfc):
                            #
                            # Quantization path for Float4E2M1FN output:
                            # 1. Compute per-vector absolute max from SwiGLU result
                            # 2. Generate scale factor C (SFC) based on max values
                            # 3. Store SFC to global memory
                            # 4. Quantize output by scaling with reciprocal of SFC
                            #
                            # Assume subtile partitioned always happens on n dimension
                            sfc_subtile_idx_mn = (
                                tile_info[0] * self.epi_tile_cnt[0] + epi_m_idx,
                                tile_info[1] * self.epi_tile_cnt[1] + out_n_idx,
                            )
                            tCgSFC = tCgSFC_mn[
                                (
                                    None,
                                    None,
                                    None,
                                    *sfc_subtile_idx_mn,
                                )
                            ]

                            #
                            # Get absolute max across a vector and Compute SFC
                            #
                            tTR_rAcc_frg = cute.logical_divide(
                                tCompute, cute.make_layout(self.sf_vec_size)
                            )
                            acc_frg = tTR_rAcc_frg.load()
                            acc_frg = epilogue_op(acc_frg)

                            # Apply element-wise absolute value using math.absf (supports vectors)
                            abs_acc_frg_ir = math.absf(acc_frg.ir_value())
                            abs_acc_frg = type(acc_frg)(
                                abs_acc_frg_ir, acc_frg.shape, acc_frg.dtype
                            )

                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = abs_acc_frg[None, vi].reduce(
                                        cute.ReductionOp.MAX,
                                        cutlass.Float32(0.0),
                                        0,  # Use 0.0 as init for abs values
                                    )
                                for vi in cutlass.range_constexpr(
                                    0, abs_acc_frg.shape[1], 2
                                ):
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ),
                                            (
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                                self.get_dtype_rcp_limits(self.c_dtype),
                                            ),
                                        )
                                    )
                                    tCrSFC_pvscale[vi], tCrSFC_pvscale[vi + 1] = (
                                        cute.arch.mul_packed_f32x2(
                                            (
                                                tCrSFC_pvscale[vi],
                                                tCrSFC_pvscale[vi + 1],
                                            ),
                                            (norm_const, norm_const),
                                        )
                                    )
                            else:
                                for vi in cutlass.range_constexpr(abs_acc_frg.shape[1]):
                                    tCrSFC_pvscale[vi] = (
                                        abs_acc_frg[None, vi].reduce(
                                            cute.ReductionOp.MAX,
                                            cutlass.Float32(0.0),
                                            0,  # Use 0.0 as init for abs values
                                        )
                                        * self.get_dtype_rcp_limits(self.c_dtype)
                                        * norm_const
                                    )

                            # TODO: need to add f32x2 -> f8x2 conversion
                            tCrSFC.store(tCrSFC_pvscale.load().to(self.sf_dtype))

                            #
                            # Store SFC to global memory
                            #
                            # TODO: Need to think about predicate on it
                            # if cute.elem_less():
                            cute.autovec_copy(tCrSFC, tCgSFC)

                            #
                            # Compute quantized output values and convert to C type
                            #
                            # TODO: need to add f8x2 -> f32x2 conversion
                            tCrSFC_qpvscale_up = tCrSFC.load().to(cutlass.Float32)
                            fp32_max = cutlass.Float32(3.40282346638528859812e38)
                            if cutlass.const_expr(self.vectorized_f32):
                                for vi in cutlass.range_constexpr(
                                    0, cute.size(tCrSFC), 2
                                ):
                                    acc_scale = cute.arch.mul_packed_f32x2(
                                        (
                                            cute.arch.rcp_approx(
                                                tCrSFC_qpvscale_up[vi]
                                            ),
                                            cute.arch.rcp_approx(
                                                tCrSFC_qpvscale_up[vi + 1]
                                            ),
                                        ),
                                        (norm_const, norm_const),
                                    )
                                    acc_scale_min0 = fmin(
                                        acc_scale[0], fp32_max, nan=True
                                    )
                                    acc_scale_min1 = fmin(
                                        acc_scale[1], fp32_max, nan=True
                                    )

                                    vec0 = tTR_rAcc_frg[None, vi]
                                    vec1 = tTR_rAcc_frg[None, vi + 1]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec0[ei], vec1[ei] = cute.arch.mul_packed_f32x2(
                                            (vec0[ei], vec1[ei]),
                                            (acc_scale_min0, acc_scale_min1),
                                        )
                            else:
                                for vi in cutlass.range_constexpr(cute.size(tCrSFC)):
                                    # TODO:Need to add E8M0 rcp approximation
                                    acc_scale = norm_const * cute.arch.rcp_approx(
                                        tCrSFC_qpvscale_up[vi]
                                    )
                                    acc_scale = fmin(acc_scale, fp32_max, nan=True)

                                    vec = tTR_rAcc_frg[None, vi]
                                    for ei in cutlass.range_constexpr(self.sf_vec_size):
                                        vec[ei] = vec[ei] * acc_scale

                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            tRS_rC.store(acc_vec.to(self.c_dtype))
                        else:
                            #
                            # Convert to C type
                            #
                            acc_vec = tiled_copy_r2s.retile(tCompute).load()
                            acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
                            tRS_rC.store(acc_vec)

                        #
                        # Store C to shared memory
                        #
                        num_prev_subtiles = num_prev_subtiles + 1
                        c_buffer = num_prev_subtiles % self.num_c_stage

                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rC,
                            tRS_sC[(None, None, None, c_buffer)],
                        )
                        # Fence and barrier to make sure shared memory store is visible to TMA store
                        cute.arch.fence_proxy(
                            "async.shared",
                            space="cta",
                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                        #
                        # TMA store C to global memory
                        #
                        if warp_idx == self.epilog_warp_id[0]:
                            cute.copy(
                                tma_atom_c,
                                bSG_sC[(None, c_buffer)],
                                bSG_gC[(None, epi_m_idx, out_n_idx)],
                            )
                            # Fence and barrier to make sure shared memory store is visible to TMA store
                            c_pipeline.producer_commit()
                            c_pipeline.producer_acquire()
                        self.epilog_sync_barrier.arrive_and_wait()

                #
                # Async arrive accumulator buffer empty
                #
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                #
                # Advance to next tile
                #
                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for idx in cutlass.range(4, unroll_full=True):
                    tile_info[idx] = sInfo[(idx, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy(
                    "async.shared",
                    space="cta",
                )
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            #
            # Dealloc the tensor memory buffer
            #
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            #
            # Wait for C store complete
            #
            c_pipeline.producer_tail()

        griddepcontrol_launch_dependents()

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for tensor memory load, then use it to partition tensor memory
        (source) and register array (destination).

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param tAcc: The accumulator tensor to be copied and partitioned
        :type tAcc: cute.Tensor
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param use_2cta_instrs: Whether use_2cta_instrs is enabled
        :type use_2cta_instrs: bool

        :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate) where:
            - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
            - tTR_tAcc: The partitioned accumulator tensor
            - tTR_rAcc_up: The partitioned accumulator tensor for acc up
            - tTR_rAcc_gate: The partitioned accumulator tensor for acc gate
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]
        """
        # Make tiledCopy for tensor memory load (Rubin uses transformed layout)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.c_layout,
            self.c_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )

        # tAcc is already transformed: (M, N, STAGE) layout
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(
            tAcc,
            epi_tile,
        )
        # (EPI_TILE_M, EPI_TILE_N)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_mnl_epi = cute.flat_divide(gC_mnl, epi_tile)

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, loopM, loopN, loopL)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)

        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_up = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        # (T2R, T2R_M, T2R_N)
        tTR_rAcc_gate = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc_up, tTR_rAcc_gate  # type: ignore[return-value]

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rC: cute.Tensor,
        tidx: cutlass.Int32,
        sC: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """
        Make tiledCopy for shared memory store, then use it to partition register
        array (source) and shared memory (destination).

        :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        :type tiled_copy_t2r: cute.TiledCopy
        :param tTR_rC: The partitioned accumulator tensor
        :type tTR_rC: cute.Tensor
        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor
        :type sepi: cute.Tensor

        :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
            - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
            - tRS_rC: The partitioned tensor C (register source)
            - tRS_sC: The partitioned tensor C (smem destination)
        :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
        """
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        # (R2S, R2S_M, R2S_N, PIPE_D)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        # (R2S, R2S_M, R2S_N)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        atom: Union[cute.CopyAtom, cute.TiledCopy],
        gC_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        sC: cute.Tensor,
    ) -> Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]:
        """Make tiledCopy for global memory store, then use it to:
        - partition register array (source) and global memory (destination) for none TMA store version;
        - partition shared memory (source) and global memory (destination) for TMA store version.

        :param tidx: The thread index in epilogue warp groups
        :type tidx: cutlass.Int32
        :param atom: The copy_atom_c to be used for TMA store version, or tiled_copy_t2r for none TMA store version
        :type atom: cute.CopyAtom or cute.TiledCopy
        :param gC_mnl: The global tensor C
        :type gC_mnl: cute.Tensor
        :param epi_tile: The epilogue tiler
        :type epi_tile: cute.Tile
        :param sC: The shared memory tensor to be copied and partitioned
        :type sC: cute.Tensor

        :return: A tuple containing :
            - For TMA store: (tma_atom_c, bSG_sC, bSG_gC) where:
                - tma_atom_c: The TMA copy atom
                - bSG_sC: The partitioned shared memory tensor C
                - bSG_gC: The partitioned global tensor C
        :rtype: Tuple[cute.CopyAtom, cute.Tensor, cute.Tensor]
        """
        # gC_mnl is already transformed: (M, N_half, loopM, loopN, loopL)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, loopM, loopN, loopL)
        gC_epi = cute.flat_divide(gC_mnl, epi_tile)
        tma_atom_c = atom
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        # ((ATOM_V, REST_V), EPI_M, EPI_N, loopM, loopN, loopL)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC

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
        num_smem_capacity: int,
        occupancy: int,
        with_breuse: bool = False,
    ) -> Tuple[int, int, int]:
        """Computes the number of stages for A/B/C operands based on heuristics.

        :param tiled_mma: The tiled MMA object defining the core computation.
        :type tiled_mma: cute.TiledMma
        :param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
        :type mma_tiler_mnk: tuple[int, int, int]
        :param a_dtype: Data type of operand A.
        :type a_dtype: type[cutlass.Numeric]
        :param b_dtype: Data type of operand B.
        :type b_dtype: type[cutlass.Numeric]
        :param epi_tile: The epilogue tile shape.
        :type epi_tile: cute.Tile
        :param c_dtype: Data type of operand C (output).
        :type c_dtype: type[cutlass.Numeric]
        :param c_layout: Layout of operand C.
        :type c_layout: utils.LayoutEnum
        :param sf_dtype: Data type of scale factor.
        :type sf_dtype: type[cutlass.Numeric]
        :param sf_vec_size: Vector size of scale factor.
        :type sf_vec_size: int
        :param num_smem_capacity: Total available shared memory capacity in bytes.
        :type num_smem_capacity: int
        :param occupancy: Target number of CTAs per SM (occupancy).
        :type occupancy: int

        :return: A tuple containing the computed number of stages for:
                 (ACC stages, A/B operand stages, C stages)
        :rtype: tuple[int, int, int]
        """
        # Default ACC stages
        num_acc_stage = 1 if (with_breuse and mma_tiler_mnk[1] in {192, 256}) else 2

        # Default C stages
        num_c_stage = 2

        # Default Tile info stages
        num_tile_stage = 2

        # Calculate smem layout and size for one stage of A, B, and C
        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,  # a tmp 1 stage is provided
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,  # a tmp 1 stage is provided
        )

        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,  # a tmp 1 stage is provided
        )

        c_smem_layout_staged_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        # 1024B alignment
        mbar_helpers_bytes = 1024
        c_bytes_per_stage = cute.size_in_bytes(c_dtype, c_smem_layout_staged_one)
        c_bytes = c_bytes_per_stage * num_c_stage

        # Calculate A/B stages:
        # Start with total smem per CTA (capacity / occupancy)
        # Subtract reserved bytes and initial C stages bytes
        # Divide remaining by bytes needed per A/B stage
        num_ab_stage = (
            num_smem_capacity // occupancy - (mbar_helpers_bytes + c_bytes)
        ) // ab_bytes_per_stage

        # Refine epilogue stages:
        # Calculate remaining smem after allocating for A/B stages and reserved bytes
        # Add remaining unused smem to epilogue
        num_c_stage += (
            num_smem_capacity
            - occupancy * ab_bytes_per_stage * num_ab_stage
            - occupancy * (mbar_helpers_bytes + c_bytes)
        ) // (occupancy * c_bytes_per_stage)
        return num_acc_stage, num_ab_stage, num_c_stage, num_tile_stage  # type: ignore[return-value]

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
        raster_along_m: bool = False,
    ) -> Tuple[utils.PersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Use persistent tile scheduler to compute the grid size for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters.
        :type max_active_clusters: cutlass.Constexpr

        :return: A tuple containing:
            - tile_sched_params: Parameters for the persistent tile scheduler.
            - grid: Grid shape for kernel launch.
        :rtype: Tuple[utils.PersistentTileSchedulerParams, tuple[int, int, int]]
        """
        c_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gc = cute.zipped_divide(c, tiler=c_shape)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl, cluster_shape_mnl, raster_along_m=raster_along_m
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def _get_tma_atom_kind(
        atom_sm_cnt: cutlass.Int32, mcast: cutlass.Boolean
    ) -> Union[
        cpasync.CopyBulkTensorTileG2SMulticastOp, cpasync.CopyBulkTensorTileG2SOp
    ]:
        """
        Select the appropriate TMA copy atom based on the number of SMs and the multicast flag.

        :param atom_sm_cnt: The number of SMs
        :type atom_sm_cnt: cutlass.Int32
        :param mcast: The multicast flag
        :type mcast: cutlass.Boolean

        :return: The appropriate TMA copy atom kind
        :rtype: cpasync.CopyBulkTensorTileG2SMulticastOp or cpasync.CopyBulkTensorTileG2SOp

        :raise ValueError: If the atom_sm_cnt is invalid
        """
        if atom_sm_cnt == 2 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 2 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
        elif atom_sm_cnt == 1 and mcast:
            return cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.ONE)
        elif atom_sm_cnt == 1 and not mcast:
            return cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        raise ValueError(f"Invalid atom_sm_cnt: {atom_sm_cnt} and {mcast}")

    @staticmethod
    def get_dtype_rcp_limits(dtype: Type[cutlass.Numeric]) -> float:
        """
        Calculates the reciprocal of the maximum absolute value for a given data type.

        :param dtype: Data type
        :type dtype: Type[cutlass.Numeric]

        :return: An float representing the reciprocal of the maximum absolute value
        :rtype: float
        """
        if dtype == cutlass.Float4E2M1FN:
            return 1 / 6.0
        if dtype == cutlass.Float8E4M3FN:
            return 1 / 448.0
        if dtype == cutlass.Float8E5M2:
            return 1 / 128.0
        return 1.0

    @staticmethod
    def is_valid_dtypes_and_scale_factor_vec_size(
        ab_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
    ) -> bool:
        """
        Check if the dtypes are valid

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]

        :return: True if the dtypes are valid, False otherwise
        :rtype: bool
        """
        is_valid = True
        if ab_dtype not in {
            cutlass.Float4E2M1FN,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
        }:
            is_valid = False

        # Check valid sf_vec_size
        if sf_vec_size not in {16, 32}:
            is_valid = False

        # Check valid sf_dtype
        if sf_dtype not in {cutlass.Float8E8M0FNU, cutlass.Float8E4M3FN}:
            is_valid = False

        # Check valid sf_dtype and sf_vec_size combinations
        if sf_dtype == cutlass.Float8E4M3FN and sf_vec_size == 32:
            is_valid = False
        if ab_dtype in {cutlass.Float8E5M2, cutlass.Float8E4M3FN} and sf_vec_size == 16:
            is_valid = False

        # Check valid c_dtype
        if c_dtype not in {
            cutlass.Float32,
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E5M2,
            cutlass.Float8E4M3FN,
            cutlass.Float4E2M1FN,
        }:
            is_valid = False

        return is_valid

    @staticmethod
    def is_valid_layouts(
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if layouts and dtypes are valid combinations

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major dimension of the A tensor
        :type a_major: str
        :param b_major: The major dimension of the B tensor
        :type b_major: str
        :param c_major: The major dimension of the C tensor
        :type c_major: str

        :return: True if the layouts are valid, False otherwise
        :rtype: bool
        """
        is_valid = True

        if ab_dtype is cutlass.Float4E2M1FN and not (a_major == "k" and b_major == "k"):
            is_valid = False
        if c_dtype is cutlass.Float4E2M1FN and c_major == "m":
            is_valid = False
        return is_valid

    @staticmethod
    def is_valid_mma_tiler_and_cluster_shape(
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> bool:
        """Check if the mma tiler and cluster shape are valid."""
        # Check valid mma_inst_shape
        if mma_inst_shape[0] not in [128, 256]:
            return False
        # SwiGLU Fusion requires even epi_tile counts
        if mma_inst_shape[1] not in [128, 256]:
            return False

        # Check valid mma_tiler
        if mma_tiler[0] not in [128, 256, 512]:
            return False
        if mma_tiler[1] not in [128, 256]:
            return False

        # Check MMA tiler vs MMA instruction relationship
        # mma_tiler[0] == mma_inst_shape[0] (no B-reuse) or 2 * mma_inst_shape[0] (B-reuse)
        if mma_tiler[0] not in (mma_inst_shape[0], 2 * mma_inst_shape[0]):
            return False
        if mma_tiler[1] != mma_inst_shape[1]:
            return False

        # Check K-dimension constraints based on data type
        if a_dtype in {cutlass.Float8E4M3FN, cutlass.Float8E5M2} and b_dtype in {
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        }:
            if mma_tiler[2] != 128 or mma_inst_shape[2] != 64:
                return False
        else:
            if mma_tiler[2] != 256 or mma_inst_shape[2] != 128:
                return False

        # Check 2CTA cluster shape constraint
        if cluster_shape_mn[0] % (2 if mma_inst_shape[0] == 256 else 1) != 0:
            return False

        # Check cluster shape validity
        def _is_power_of_2(x):
            return x > 0 and (x & (x - 1)) == 0

        if (
            cluster_shape_mn[0] * cluster_shape_mn[1] > 16
            or cluster_shape_mn[0] <= 0
            or cluster_shape_mn[1] <= 0
            or cluster_shape_mn[0] > 4
            or cluster_shape_mn[1] > 4
            or not _is_power_of_2(cluster_shape_mn[0])
            or not _is_power_of_2(cluster_shape_mn[1])
        ):
            return False

        # We only support cluster shape n = 1 for now
        if cluster_shape_mn[1] != 1:
            return False
        return True

    @staticmethod
    def is_valid_tensor_alignment(
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the tensor alignment is valid

        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the problem shape is valid, False otherwise
        :rtype: bool
        """
        is_valid = True

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
            is_valid = False
        return is_valid

    @classmethod
    def can_implement(
        cls,
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_dtype: Type[cutlass.Numeric],
        mma_inst_shape: Tuple[int, int, int],
        mma_tiler: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        a_major: str,
        b_major: str,
        c_major: str,
    ) -> bool:
        """
        Check if the gemm can be implemented

        :param ab_dtype: The data type of the A and B operands
        :type ab_dtype: Type[cutlass.Numeric]
        :param sf_dtype: The data type of the scale factor
        :type sf_dtype: Type[cutlass.Numeric]
        :param sf_vec_size: The vector size of the scale factor
        :type sf_vec_size: int
        :param c_dtype: The data type of the output tensor
        :type c_dtype: Type[cutlass.Numeric]
        :param mma_tiler_mn: The (M, N) shape of the MMA instruction tiler
        :type mma_tiler_mn: Tuple[int, int]
        :param cluster_shape_mn: The (ClusterM, ClusterN) shape of the CTA cluster
        :type cluster_shape_mn: Tuple[int, int]
        :param m: The number of rows in the A tensor
        :type m: cutlass.Int64
        :param n: The number of columns in the B tensor
        :type n: cutlass.Int64
        :param k: The number of columns in the A tensor
        :type k: cutlass.Int64
        :param l: The number of columns in the C tensor
        :type l: cutlass.Int64
        :param a_major: The major axis of the A tensor
        :type a_major: str
        :param b_major: The major axis of the B tensor
        :type b_major: str
        :param c_major: The major axis of the C tensor
        :type c_major: str

        :return: True if the gemm can be implemented, False otherwise
        :rtype: bool
        """
        # Check data types
        if not cls.is_valid_dtypes_and_scale_factor_vec_size(
            a_dtype, sf_dtype, sf_vec_size, c_dtype
        ):
            return False

        # Check layouts
        if not cls.is_valid_layouts(a_dtype, c_dtype, a_major, b_major, c_major):
            return False

        # Check MMA tiler and cluster shape
        if not cls.is_valid_mma_tiler_and_cluster_shape(
            a_dtype, b_dtype, mma_inst_shape, mma_tiler, cluster_shape_mn
        ):
            return False

        # Check tensor alignment
        if not cls.is_valid_tensor_alignment(
            m, n, k, l, a_dtype, c_dtype, a_major, b_major, c_major
        ):
            return False

        # Check A/B layout
        if not (a_major == "k" and b_major == "k"):
            return False
        return True

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        c_sf_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        scale_k = k // scaling_vector_size
        interm_size = n // 2
        num_tiles = m // tile_size
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        a_sf = cute.make_tensor(
            a_sf_ptr,
            layout=cute.make_ordered_layout((orig_m, scale_k, 1), order=(1, 0, 2)),
        )
        b_sf = cute.make_tensor(
            b_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, n // 128, 4, scale_k // 4, l), order=(2, 1, 4, 0, 3, 5)
            ),
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, interm_size, 1), order=(1, 0, 2))
        )
        c_sf = cute.make_tensor(
            c_sf_ptr,
            layout=cute.make_ordered_layout(
                (32, 4, m // 128, 4, interm_size // (scaling_vector_size * 4), l),
                order=(2, 1, 4, 0, 3, 5),
            ),
        )
        alpha = cute.make_tensor(alpha_ptr, layout=cute.make_layout((l,)))

        tile_idx_to_group_idx = cute.make_tensor(
            tile_idx_to_group_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        global_sf = cute.make_tensor(global_sf_ptr, layout=cute.make_layout((1,)))

        return self(
            a,
            b,
            c,
            a_sf,
            b_sf,
            c_sf,
            global_sf,
            tile_idx_to_group_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            max_active_clusters=max_active_clusters,
            stream=stream,
            epilogue_op=epilogue_op,
        )


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


@cute.jit
def cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
    sf_swizzled_tensor: cute.Tensor,
    sf_unswizzled_tensor: cute.Tensor,
):
    """Convert scale factor tensor from mma specification M(32x4xrest_m)xK(4xrest_k)xL layout to MKL layout"""
    # sf_swizzled_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 0, 3)
    sf_swizzled_tensor = cute.group_modes(sf_swizzled_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_unswizzled_tensor)):
        mkl_coord = sf_unswizzled_tensor.layout.get_hier_coord(i)
        sf_unswizzled_tensor[mkl_coord] = sf_swizzled_tensor[mkl_coord]


# ============================================================================
# Run utilities
# ============================================================================


def create_mask(group_m_list, mma_tiler_m, permuted_m=None):
    """Create mask and group mapping for contiguous grouped GEMM with gather and SwiGLU.

    :param group_m_list: List of M values for each group (will be aligned to mma_tiler_m)
    :param mma_tiler_m: MMA tile size in M dimension, also used for alignment
    :param permuted_m: Optional padded M dimension for cuda_graph support
    :return: Tuple of (valid_m, aligned_group_m_list, tile_idx_to_expert_idx,
             tile_idx_to_mn_limit, num_non_exiting_tiles)
    """
    valid_m = 0
    aligned_group_m_list = []
    tile_idx_to_expert_idx = []
    tile_idx_to_mn_limit = []

    for i, group_m in enumerate(group_m_list):
        aligned_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        aligned_group_m_list.append(aligned_group_m)

        num_tiles_in_group = aligned_group_m // mma_tiler_m
        tile_idx_to_expert_idx.extend([i] * num_tiles_in_group)
        for tile_idx_in_group in range(num_tiles_in_group):
            tile_idx_to_mn_limit.append(
                valid_m + min(tile_idx_in_group * mma_tiler_m + mma_tiler_m, group_m)
            )
        valid_m += aligned_group_m

    num_non_exiting_tiles = len(tile_idx_to_expert_idx)

    if permuted_m is not None:
        if permuted_m < valid_m:
            raise ValueError(
                f"permuted_m ({permuted_m}) must be >= valid_m ({valid_m})."
            )
        if permuted_m > valid_m:
            num_padding_tiles = (permuted_m - valid_m) // mma_tiler_m
            tile_idx_to_expert_idx.extend([int(-2e9)] * num_padding_tiles)
            tile_idx_to_mn_limit.extend([int(-2e9)] * num_padding_tiles)

    tile_idx_to_expert_idx = torch.tensor(
        tile_idx_to_expert_idx, device="cuda", dtype=torch.int32
    )
    num_non_exiting_tiles_tensor = torch.tensor(
        [num_non_exiting_tiles], device="cuda", dtype=torch.int32
    )
    tile_idx_to_mn_limit_tensor = torch.tensor(
        tile_idx_to_mn_limit, device="cuda", dtype=torch.int32
    )

    return (
        valid_m,
        aligned_group_m_list,
        tile_idx_to_expert_idx,
        num_non_exiting_tiles_tensor,
        tile_idx_to_mn_limit_tensor,
    )


def create_scale_factor_tensor(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    ref_shape = (num_groups, mn, sf_k)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    ref_permute_order = (1, 2, 0)
    mma_permute_order = (3, 4, 1, 5, 2, 0)

    ref_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape,
        torch.float32,
        permute_order=ref_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )

    cute_f32_torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )

    cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
        from_dlpack(ref_f32_torch_tensor_cpu),
        from_dlpack(cute_f32_torch_tensor_cpu),
    )

    cute_f32_torch_tensor = cute_f32_torch_tensor_cpu.cuda()

    ref_f32_torch_tensor_cpu = (
        ref_f32_torch_tensor_cpu.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
        .permute(*ref_permute_order)
    )
    ref_f32_torch_tensor_cpu = ref_f32_torch_tensor_cpu[:, :k, :]

    cute_tensor, cute_torch_tensor = cutlass_torch.cute_tensor_like(
        cute_f32_torch_tensor_cpu,
        dtype,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    cute_tensor = cutlass_torch.convert_cute_tensor(
        cute_f32_torch_tensor,
        cute_tensor,
        dtype,
        is_dynamic_layout=True,
    )
    return ref_f32_torch_tensor_cpu, cute_tensor, cute_torch_tensor


def create_scale_factor_tensor_unswizzled(num_groups, mn, k, sf_vec_size, dtype):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(k, sf_vec_size)
    sf_ref = cutlass_torch.matrix(
        num_groups,
        mn,
        sf_k,
        False,
        cutlass.Float32,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )

    sf_tensor, sf_torch = cutlass_torch.cute_tensor_like(
        sf_ref, dtype, is_dynamic_layout=True, assumed_align=16
    )

    sf_ref = (
        sf_ref.permute(2, 0, 1)
        .unsqueeze(-1)
        .expand(num_groups, mn, sf_k, sf_vec_size)
        .reshape(num_groups, mn, sf_k * sf_vec_size)
        .permute(1, 2, 0)
    )
    sf_ref = sf_ref[:, :k, :]
    return sf_ref, sf_tensor, sf_torch


def create_sf_layout_tensor(num_groups, mn, nk, sf_vec_size):
    def ceil_div(a, b):
        return (a + b - 1) // b

    sf_k = ceil_div(nk, sf_vec_size)

    atom_m = (32, 4)
    atom_k = 4
    mma_shape = (
        num_groups,
        ceil_div(mn, atom_m[0] * atom_m[1]),
        ceil_div(sf_k, atom_k),
        atom_m[0],
        atom_m[1],
        atom_k,
    )

    mma_permute_order = (3, 4, 1, 5, 2, 0)

    cute_f32_torch_tensor = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape,
        torch.float32,
        permute_order=mma_permute_order,
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )
    return cute_f32_torch_tensor, sf_k


def create_token_id_mapping_tensor(
    group_m_list, mma_tiler_m, max_token_id, permuted_m=None
):
    """Create token_id_mapping tensor for gather operation with random distribution."""
    valid_m = 0
    for group_m in group_m_list:
        valid_m += ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m

    tensor_m = permuted_m if permuted_m is not None else valid_m

    base_data = torch.full((tensor_m,), -1, dtype=torch.int32)

    accumulated_m = 0
    for group_m in group_m_list:
        start_idx = accumulated_m
        rounded_group_m = ((group_m + mma_tiler_m - 1) // mma_tiler_m) * mma_tiler_m
        random_token_ids = torch.randint(0, max_token_id, (group_m,), dtype=torch.int32)
        base_data[start_idx : start_idx + group_m] = random_token_ids
        accumulated_m += rounded_group_m

    token_id_mapping_ref = base_data.clone()
    token_id_mapping_tensor, token_id_mapping_torch = cutlass_torch.cute_tensor_like(
        token_id_mapping_ref, cutlass.Int32, is_dynamic_layout=True, assumed_align=4
    )
    return token_id_mapping_ref, token_id_mapping_tensor, token_id_mapping_torch


def create_tensors(
    num_groups,
    group_m_list,
    n,
    k,
    a_major,
    b_major,
    cd_major,
    a_dtype,
    b_dtype,
    c_dtype,
    sf_dtype,
    sf_vec_size,
    mma_tiler_m,
    permuted_m=None,
):
    """Create tensors for contiguous grouped GEMM with gather operation and SwiGLU fusion.

    Output C has N/2 columns since SwiGLU combines pairs of (up, gate) from interleaved B weights.
    """
    torch.manual_seed(1111)

    alpha_torch_cpu = torch.randn((num_groups,), dtype=torch.float32)

    (
        valid_m,
        aligned_group_m_list,
        _tile_idx_to_expert_idx,
        _num_non_exiting_tiles,
        _tile_idx_to_mn_limit,
    ) = create_mask(group_m_list, mma_tiler_m, permuted_m)

    max_m = max(group_m_list)

    tensor_m = permuted_m if permuted_m is not None else valid_m

    a_torch_cpu = cutlass_torch.matrix(1, max_m, k, a_major == "m", cutlass.Float32)
    b_torch_cpu = cutlass_torch.matrix(
        num_groups, n, k, b_major == "n", cutlass.Float32
    )
    c_torch_cpu = cutlass_torch.matrix(
        1, tensor_m, n // 2, cd_major == "m", cutlass.Float32
    )

    a_tensor, a_torch_gpu = cutlass_torch.cute_tensor_like(
        a_torch_cpu, a_dtype, is_dynamic_layout=True, assumed_align=16
    )
    b_tensor, b_torch_gpu = cutlass_torch.cute_tensor_like(
        b_torch_cpu, b_dtype, is_dynamic_layout=True, assumed_align=16
    )
    c_tensor, c_torch_gpu = cutlass_torch.cute_tensor_like(
        c_torch_cpu, c_dtype, is_dynamic_layout=True, assumed_align=16
    )

    a_tensor.mark_compact_shape_dynamic(
        mode=1 if a_major == "k" else 0,
        stride_order=(2, 0, 1) if a_major == "k" else (2, 1, 0),
        divisibility=32 if a_dtype == cutlass.Float4E2M1FN else 16,
    )
    b_tensor.mark_compact_shape_dynamic(
        mode=1 if b_major == "k" else 0,
        stride_order=(2, 0, 1) if b_major == "k" else (2, 1, 0),
        divisibility=32 if b_dtype == cutlass.Float4E2M1FN else 16,
    )
    c_tensor.mark_compact_shape_dynamic(
        mode=1 if cd_major == "n" else 0,
        stride_order=(2, 0, 1) if cd_major == "n" else (2, 1, 0),
        divisibility=32 if c_dtype == cutlass.Float4E2M1FN else 16,
    )

    sfa_torch_cpu, sfa_tensor, sfa_torch_gpu = create_scale_factor_tensor_unswizzled(
        1, max_m, k, sf_vec_size, sf_dtype
    )
    sfb_torch_cpu, sfb_tensor, sfb_torch_gpu = create_scale_factor_tensor(
        num_groups, n, k, sf_vec_size, sf_dtype
    )

    token_id_mapping_cpu, token_id_mapping, token_id_mapping_torch = (
        create_token_id_mapping_tensor(
            group_m_list, mma_tiler_m, max_token_id=max_m, permuted_m=permuted_m
        )
    )

    tile_idx_to_expert_idx = from_dlpack(_tile_idx_to_expert_idx).mark_layout_dynamic()
    tile_idx_to_mn_limit = from_dlpack(_tile_idx_to_mn_limit).mark_layout_dynamic()
    num_non_exiting_tiles = from_dlpack(_num_non_exiting_tiles).mark_layout_dynamic()

    alpha = from_dlpack(alpha_torch_cpu.cuda()).mark_layout_dynamic()

    # Create sfc_tensor and norm_const_tensor when c_dtype is Float4E2M1FN
    sfc_torch_cpu = None
    sfc_tensor = None
    sfc_torch_gpu = None
    norm_const_torch_cpu = None
    norm_const_tensor = None
    norm_const_torch_gpu = None
    n_out = n // 2
    if c_dtype == cutlass.Float4E2M1FN:
        sfc_torch_cpu, sfc_tensor, sfc_torch_gpu = create_scale_factor_tensor(
            1, tensor_m, n_out, sf_vec_size, sf_dtype
        )
        norm_const_torch = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        norm_const_tensor = from_dlpack(norm_const_torch).mark_layout_dynamic()
        norm_const_torch_cpu = norm_const_torch.cpu()

    return (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    )


def run(
    nkl: Tuple[int, int, int],
    group_m_list: Tuple[int, ...],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sf_vec_size: int,
    a_major: str,
    b_major: str,
    c_major: str,
    mma_inst_shape: Tuple[int, int, int],
    mma_tiler: Tuple[int, int, int],
    cluster_shape_mn: Tuple[int, int],
    tolerance: float,
    warmup_iterations: int = 0,
    iterations: int = 1,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    permuted_m: int = None,
    raster_along_m: bool = False,
    **kwargs,
):
    """Run contiguous grouped GEMM with gather and SwiGLU fusion on Rubin."""
    mma_tiler_m = mma_tiler[0]

    print(
        "Running Rubin Persistent Contiguous Grouped GEMM with Gather and SwiGLU Fusion:"
    )
    print(f"nkl: {nkl}")
    print(f"group_m_list: {group_m_list}")
    print(
        f"A dtype: {a_dtype}, B dtype: {b_dtype}, C dtype: {c_dtype}, "
        f"SF dtype: {sf_dtype}, SF Vec size: {sf_vec_size}"
    )
    if permuted_m is not None:
        print(f"Padded M (CUDA graph support): {permuted_m}")
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"MMA Inst Shape: {mma_inst_shape}, MMA Tiler: {mma_tiler}")
    print(f"Cluster Shape: {cluster_shape_mn}")
    print(f"Raster along M: {raster_along_m}")

    n, k, num_groups = nkl

    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required to run this example!")

    if not Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel.can_implement(
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        sf_dtype=sf_dtype,
        sf_vec_size=sf_vec_size,
        c_dtype=c_dtype,
        mma_inst_shape=mma_inst_shape,
        mma_tiler=mma_tiler,
        cluster_shape_mn=cluster_shape_mn,
        m=mma_tiler_m,
        n=n,
        k=k,
        l=num_groups,
        a_major=a_major,
        b_major=b_major,
        c_major=c_major,
    ):
        raise TypeError(
            f"Unsupported testcase a_dtype={a_dtype}, b_dtype={b_dtype}, sf_dtype={sf_dtype}, "
            f"sf_vec_size={sf_vec_size}, c_dtype={c_dtype}, mma_inst_shape={mma_inst_shape}, "
            f"mma_tiler={mma_tiler}, cluster_shape_mn={cluster_shape_mn}"
        )

    (
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        a_torch_cpu,
        b_torch_cpu,
        c_torch_cpu,
        sfa_torch_cpu,
        sfb_torch_cpu,
        sfc_torch_cpu,
        norm_const_torch_cpu,
        alpha_torch_cpu,
        a_torch_gpu,
        b_torch_gpu,
        c_torch_gpu,
        sfa_torch_gpu,
        sfb_torch_gpu,
        sfc_torch_gpu,
        norm_const_torch_gpu,
        aligned_group_m_list,
        valid_m,
        token_id_mapping_cpu,
    ) = create_tensors(
        num_groups,
        group_m_list,
        n,
        k,
        a_major,
        b_major,
        c_major,
        a_dtype,
        b_dtype,
        c_dtype,
        sf_dtype,
        sf_vec_size,
        mma_tiler_m,
        permuted_m,
    )

    gemm = Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel(
        sf_vec_size,
        mma_inst_shape,
        mma_tiler,
        cluster_shape_mn,
        True,
        topk=1,
        raster_along_m=raster_along_m,
    )

    hardware_info = cutlass.utils.HardwareInfo()
    max_active_clusters = hardware_info.get_max_active_clusters(
        cluster_shape_mn[0] * cluster_shape_mn[1]
    )

    torch_stream = torch.cuda.current_stream()
    current_stream = cuda.CUstream(torch_stream.cuda_stream)

    compiled_gemm = cute.compile(
        gemm,
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        max_active_clusters,
        current_stream,
    )

    compiled_gemm(
        a_tensor,
        b_tensor,
        c_tensor,
        sfa_tensor,
        sfb_tensor,
        sfc_tensor,
        norm_const_tensor,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        token_id_mapping,
        num_non_exiting_tiles,
        alpha,
        current_stream,
    )

    torch.cuda.synchronize()

    if not skip_ref_check:
        print("Verifying results...")
        interleave_granularity = 64
        n_out = n // 2

        # Step 1: Compute full GEMM
        gemm_result = torch.empty((1, valid_m, n), dtype=torch.float32)
        start = 0
        a_torch_cpu_f32 = torch.einsum(
            "mk,mk->mk", a_torch_cpu[:, :, 0], sfa_torch_cpu[:, :, 0]
        )
        for i, group_m in enumerate(aligned_group_m_list):
            end = start + group_m
            res_a = a_torch_cpu_f32[token_id_mapping_cpu[start:end]]
            res_b = torch.einsum(
                "nk,nk->nk", b_torch_cpu[:, :, i], sfb_torch_cpu[:, :, i]
            )
            gemm_result[0, start:end, :] = (
                torch.einsum("mk,nk->mn", res_a, res_b) * alpha_torch_cpu[i]
            )
            start = end

        # Step 2: Apply SwiGLU on interleaved GEMM result
        assert n % (2 * interleave_granularity) == 0
        ref = torch.empty((1, valid_m, n_out), dtype=torch.float32)
        for n_block in range(0, n, 2 * interleave_granularity):
            up_result = gemm_result[0, :, n_block : n_block + interleave_granularity]
            gate_result = gemm_result[
                0,
                :,
                n_block + interleave_granularity : n_block + 2 * interleave_granularity,
            ]
            silu_gate = gate_result * torch.sigmoid(gate_result)
            output_block = up_result * silu_gate
            out_start = n_block // 2
            out_end = out_start + interleave_granularity
            ref[0, :, out_start:out_end] = output_block

        ref = ref.permute((1, 2, 0))

        # Convert c back to f32 for comparison
        res = c_torch_cpu.cuda()
        cute.testing.convert(
            c_tensor,
            from_dlpack(res, assumed_align=16).mark_layout_dynamic(
                leading_dim=(1 if c_major == "n" else 0)
            ),
        )

        res = res[:valid_m]
        mask = token_id_mapping_cpu[:valid_m] >= 0
        res = res.cpu()[mask]
        ref = ref[mask]

        print(f"valid_m: {valid_m}, ref.shape: {ref.shape}, res.shape: {res.shape}")

        if c_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
            torch.testing.assert_close(res.cpu(), ref.cpu(), atol=tolerance, rtol=1e-02)
        elif c_dtype in (cutlass.Float8E5M2, cutlass.Float8E4M3FN):
            ref_f8_ = torch.empty(
                *(1, valid_m, n_out), dtype=torch.uint8, device="cuda"
            ).permute(1, 2, 0)
            ref_f8 = from_dlpack(ref_f8_, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            ref_f8.element_type = c_dtype
            ref_device = ref.cuda()
            ref_tensor = from_dlpack(ref_device, assumed_align=16).mark_layout_dynamic(
                leading_dim=1
            )
            cute.testing.convert(ref_tensor, ref_f8)
            cute.testing.convert(ref_f8, ref_tensor)
            torch.testing.assert_close(
                res.cpu(), ref_device.cpu(), atol=tolerance, rtol=1e-02
            )
        elif c_dtype is cutlass.Float4E2M1FN:

            def ceil_div(a, b):
                return (a + b - 1) // b

            def simulate_f8_quantization(tensor_f32, f8_dtype):
                shape = tensor_f32.shape
                f8_torch = torch.empty(*shape, dtype=torch.uint8, device="cuda")
                f8_tensor = from_dlpack(f8_torch, assumed_align=16).mark_layout_dynamic(
                    leading_dim=1
                )
                f8_tensor.element_type = f8_dtype
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(
                    f32_device, assumed_align=16
                ).mark_layout_dynamic(leading_dim=1)
                cute.testing.convert(f32_tensor, f8_tensor)
                cute.testing.convert(f8_tensor, f32_tensor)
                return f32_device.cpu()

            def simulate_nvfp4_quantization(tensor_f32):
                m_dim, n_dim, ng = tensor_f32.shape
                ref_f32_torch = cutlass_torch.matrix(
                    ng, m_dim, n_dim, False, cutlass.Float32
                )
                f4_tensor, _ = cutlass_torch.cute_tensor_like(
                    ref_f32_torch,
                    cutlass.Float4E2M1FN,
                    is_dynamic_layout=True,
                    assumed_align=16,
                )
                f32_device = tensor_f32.cuda()
                f32_tensor = from_dlpack(
                    f32_device, assumed_align=16
                ).mark_layout_dynamic(leading_dim=1)
                cute.testing.convert(f32_tensor, f4_tensor)
                cute.testing.convert(f4_tensor, f32_tensor)
                return f32_device.cpu()

            def compute_scale_factor(
                tensor_f32, sf_vec_size_local, norm_const, rcp_limits
            ):
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = ceil_div(n_dim, sf_vec_size_local)
                padded_n = sfn * sf_vec_size_local
                if padded_n > n_dim:
                    tensor_padded = torch.zeros(
                        m_dim, padded_n, ng, dtype=tensor_f32.dtype
                    )
                    tensor_padded[:, :n_dim, :] = tensor_f32
                else:
                    tensor_padded = tensor_f32
                tensor_reshaped = tensor_padded.view(m_dim, sfn, sf_vec_size_local, ng)
                abs_max, _ = torch.abs(tensor_reshaped).max(dim=2)
                scale_factor = abs_max * norm_const * rcp_limits
                return scale_factor

            def apply_quantization_scale(
                tensor_f32, scale_factor, sf_vec_size_local, norm_const
            ):
                m_dim, n_dim, ng = tensor_f32.shape
                sfn = scale_factor.shape[1]
                fp32_max = torch.tensor(3.40282346638528859812e38, dtype=torch.float32)
                scale_rcp = norm_const * scale_factor.reciprocal()
                scale_rcp = torch.where(torch.isinf(scale_rcp), fp32_max, scale_rcp)
                scale_rcp_expanded = scale_rcp.unsqueeze(2).expand(
                    m_dim, sfn, sf_vec_size_local, ng
                )
                scale_rcp_expanded = scale_rcp_expanded.reshape(
                    m_dim, sfn * sf_vec_size_local, ng
                )
                scale_rcp_expanded = scale_rcp_expanded[:, :n_dim, :]
                return tensor_f32 * scale_rcp_expanded

            def unswizzle_kernel_sfc(
                sfc_tensor_local, permuted_m_local, n_out_local, sf_vec_size_local
            ):
                sfn = ceil_div(n_out_local, sf_vec_size_local)
                unswizzled_sfc = torch.empty(
                    permuted_m_local, sfn, 1, dtype=torch.float32
                )
                swizzled_sfc_cpu, _ = create_sf_layout_tensor(
                    1, permuted_m_local, n_out_local, sf_vec_size_local
                )
                swizzled_sfc_tensor, swizzled_sfc_torch = (
                    cutlass_torch.cute_tensor_like(
                        swizzled_sfc_cpu,
                        cutlass.Float32,
                        is_dynamic_layout=True,
                        assumed_align=16,
                    )
                )
                cute.testing.convert(sfc_tensor_local, swizzled_sfc_tensor)
                swizzled_sfc_cpu = swizzled_sfc_torch.cpu()
                cvt_sf_M32x4xrm_K4xrk_L_to_MKL(
                    from_dlpack(swizzled_sfc_cpu),
                    from_dlpack(unswizzled_sfc),
                )
                return unswizzled_sfc

            norm_const = norm_const_torch_cpu.item()
            rcp_limits = gemm.get_dtype_rcp_limits(c_dtype)

            ref_sfc_f32 = compute_scale_factor(ref, sf_vec_size, norm_const, rcp_limits)
            ref_sfc_f32 = simulate_f8_quantization(ref_sfc_f32, sf_dtype)

            permuted_m_val = token_id_mapping_cpu.shape[0]
            kernel_sfc = unswizzle_kernel_sfc(
                sfc_tensor, permuted_m_val, n_out, sf_vec_size
            )
            torch.testing.assert_close(
                ref_sfc_f32, kernel_sfc[:valid_m][mask], atol=tolerance, rtol=1e-02
            )
            print("SFC Tensor comparison passed!")

            ref_scaled = apply_quantization_scale(
                ref, ref_sfc_f32, sf_vec_size, norm_const
            )
            ref_quantized = simulate_nvfp4_quantization(ref_scaled)

            print("Verifying C Tensor...")
            res_cpu = res.cpu()
            diff = torch.abs(res_cpu - ref_quantized)
            within_tolerance = (diff <= tolerance) | (
                diff <= torch.abs(ref_quantized) * 1e-02
            )
            pass_rate = within_tolerance.float().mean().item()
            print(f"C Tensor pass rate: {pass_rate * 100:.2f}% (threshold: 95%)")
            assert pass_rate >= 0.95, (
                f"Only {pass_rate * 100:.2f}% elements within tolerance, expected >= 95%"
            )

    def generate_tensors():
        (
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            *_,
        ) = create_tensors(
            num_groups,
            group_m_list,
            n,
            k,
            a_major,
            b_major,
            c_major,
            a_dtype,
            b_dtype,
            c_dtype,
            sf_dtype,
            sf_vec_size,
            mma_tiler_m,
            permuted_m,
        )
        return cute.testing.JitArguments(
            a_tensor,
            b_tensor,
            c_tensor,
            sfa_tensor,
            sfb_tensor,
            sfc_tensor,
            norm_const_tensor,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            alpha,
            current_stream,
        )

    workspace_count = 1
    if use_cold_l2:
        tensor_m = permuted_m if permuted_m is not None else valid_m
        one_workspace_bytes = (
            a_torch_gpu.numel() * a_torch_gpu.element_size()
            + b_torch_gpu.numel() * b_torch_gpu.element_size()
            + c_torch_gpu.numel() * c_torch_gpu.element_size()
            + sfa_torch_gpu.numel() * sfa_torch_gpu.element_size()
            + sfb_torch_gpu.numel() * sfb_torch_gpu.element_size()
            + (tensor_m // mma_tiler_m) * 4
            + (tensor_m // mma_tiler_m) * 4
            + tensor_m * 4
            + 1 * 4
            + alpha_torch_cpu.numel() * alpha_torch_cpu.element_size()
        )
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    exec_time = cute.testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        stream=current_stream,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
    )

    return exec_time


def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Invalid format. Expected comma-separated integers."
        ) from None


def read_benchmark_file(
    filepath: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Read benchmark file and return nkl and group_m_list."""
    problems = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                dims = parts[1].split("x")
                if len(dims) == 3:
                    m, n, k = int(dims[0]), int(dims[1]), int(dims[2])
                    problems.append((m, n, k))

        if not problems:
            raise ValueError(f"No valid problems found in benchmark file: {filepath}")

        m_first, n, k = problems[0]
        num_groups = len(problems)
        m_values = tuple(m for m, _, _ in problems)

        print(f"Loaded {num_groups} problems from benchmark file")
        print(f"Using N={n}, K={k}, L={num_groups}")
        print(f"M values per group: {m_values}")

        return ((n, k, num_groups), m_values)

    except FileNotFoundError:
        raise argparse.ArgumentTypeError(
            f"Benchmark file not found: {filepath}"
        ) from None
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Error reading benchmark file: {e}") from None


def parse_benchmark_arg(
    arg: str,
) -> Tuple[Tuple[int, int, int], Tuple[int, ...]]:
    """Parse benchmark argument string."""
    match_list = re.match(r"\[([\d,\s]+)\]\s*x\s*(\d+)\s*x\s*(\d+)", arg)
    if match_list:
        m_str = match_list.group(1)
        n = int(match_list.group(2))
        k = int(match_list.group(3))
        try:
            m_values = tuple(int(x.strip()) for x in m_str.split(","))
            num_groups = len(m_values)
            return ((n, k, num_groups), m_values)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid integer list in benchmark argument: {arg}"
            ) from None

    parts = arg.split("x")
    if len(parts) == 4:
        try:
            m, n, k, num_groups = [int(x.strip()) for x in parts]
            m_values = tuple([m] * num_groups)
            return ((n, k, num_groups), m_values)
        except ValueError:
            pass

    raise argparse.ArgumentTypeError(
        f"Invalid benchmark argument format. Got: {arg}"
    ) from None


def main():
    """Main entry point for running the Rubin SwiGLU fusion kernel."""
    parser = argparse.ArgumentParser(
        description="Rubin BlockScaled Contiguous Gather Grouped GEMM with SwiGLU Fusion."
    )

    parser.add_argument("--nkl", type=parse_comma_separated_ints, default=(256, 512, 1))
    parser.add_argument("--fixed_m", type=int, default=None)
    parser.add_argument("--custom_mask", type=parse_comma_separated_ints, default=None)
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--permuted_m", type=int, default=None)
    parser.add_argument(
        "--mma_inst_shape", type=parse_comma_separated_ints, default=(128, 128, 128)
    )
    parser.add_argument(
        "--mma_tiler", type=parse_comma_separated_ints, default=(128, 128, 256)
    )
    parser.add_argument(
        "--cluster_shape_mn", type=parse_comma_separated_ints, default=(1, 1)
    )
    parser.add_argument("--a_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--b_dtype", type=cutlass.dtype, default=cutlass.Float4E2M1FN)
    parser.add_argument("--c_dtype", type=cutlass.dtype, default=cutlass.BFloat16)
    parser.add_argument("--sf_dtype", type=cutlass.dtype, default=cutlass.Float8E4M3FN)
    parser.add_argument("--sf_vec_size", type=int, default=16)
    parser.add_argument("--a_major", choices=["k"], type=str, default="k")
    parser.add_argument("--b_major", choices=["k"], type=str, default="k")
    parser.add_argument("--c_major", choices=["n", "m"], type=str, default="n")
    parser.add_argument("--tolerance", type=float, default=1e-01)
    parser.add_argument("--warmup_iterations", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true", default=False)
    parser.add_argument("--raster_along_m", action="store_true", default=False)

    args = parser.parse_args()

    if args.benchmark:
        if os.path.isfile(args.benchmark):
            nkl, group_m_list = read_benchmark_file(args.benchmark)
        else:
            nkl, group_m_list = parse_benchmark_arg(args.benchmark)
    else:
        if len(args.nkl) != 3:
            parser.error("--nkl must contain exactly 3 values")
        n, k, num_groups = args.nkl
        nkl = (n, k, num_groups)

        if args.custom_mask is not None:
            group_m_list = args.custom_mask
            if len(group_m_list) != num_groups:
                parser.error(f"--custom_mask must have exactly {num_groups} values")
        elif args.fixed_m is not None:
            group_m_list = tuple([args.fixed_m] * num_groups)
        else:
            group_m_list = tuple([128] * num_groups)

    if len(args.mma_inst_shape) != 3:
        parser.error("--mma_inst_shape must contain exactly 3 values")
    if len(args.mma_tiler) != 3:
        parser.error("--mma_tiler must contain exactly 3 values")
    if len(args.cluster_shape_mn) != 2:
        parser.error("--cluster_shape_mn must contain exactly 2 values")

    exec_time = run(
        nkl,
        group_m_list,
        args.a_dtype,
        args.b_dtype,
        args.c_dtype,
        args.sf_dtype,
        args.sf_vec_size,
        args.a_major,
        args.b_major,
        args.c_major,
        args.mma_inst_shape,
        args.mma_tiler,
        args.cluster_shape_mn,
        args.tolerance,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        args.permuted_m,
        args.raster_along_m,
    )
    print(f"Execution time: {exec_time:.2f} us")
    print("PASS")


if __name__ == "__main__":
    main()
