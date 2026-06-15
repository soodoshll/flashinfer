"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Auto-tuner for CuteDSL NVFP4 MoE kernels.

This module provides a TunableRunner implementation for the CuteDSL NVFP4 MoE
kernels, enabling automatic performance tuning across different GEMM tactics.

Supports two architectures:
- Blackwell (SM100): tactic = (mma_tiler_mn, cluster_shape_mn, raster_along_m)
- Rubin (SM107): tactic = (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)

Reference: TensorRT-LLM/tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py
"""

import itertools
import logging
from typing import Any, Callable, Dict, List, Tuple

import torch

from ...autotuner import (
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...utils import get_compute_capability
from ..utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ._inputs_helper import CuteDslMoEInputsHelper

logger = logging.getLogger(__name__)


# =============================================================================
# Blackwell (SM100) Tactics
# =============================================================================


def get_blackwell_gemm1_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Blackwell tactics for GEMM1 (Gather + SwiGLU Fusion).

    Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [(tile_size // 128, 1)]
    raster_along_m_candidates = [False]

    return [
        (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
            mma_tiler_mn_candidates,
            cluster_shape_mn_candidates,
            raster_along_m_candidates,
        )
    ]


def get_blackwell_gemm2_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Blackwell tactics for GEMM2 (Finalize Fusion).

    The finalize kernel's MMA shape must match tile_size because it consumes
    the upstream gemm1 output layout. At tile_size=128 it uses 1-CTA mma_m=128;
    at tile_size=256 it uses 2-CTA mma_m=256 (use_2cta_instrs=True). Returning a
    1-CTA tactic at tile_size=256 yields a layout mismatch and incorrect output
    (bug #3067, fixed upstream by #3171).

    Format: (mma_tiler_mn, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_mn_candidates = [(tile_size, 128), (tile_size, 256)]
    cluster_shape_mn_candidates = [
        (tile_size // 128, 1),
        (tile_size // 128, 2),
    ]
    raster_along_m_candidates = [False]

    return [
        (mma_tiler_mn, cluster_shape_mn, raster_along_m)
        for mma_tiler_mn, cluster_shape_mn, raster_along_m in itertools.product(
            mma_tiler_mn_candidates,
            cluster_shape_mn_candidates,
            raster_along_m_candidates,
        )
    ]


def get_blackwell_moe_valid_tactics() -> List[Tuple]:
    """Get all valid Blackwell MoE tactic combinations.

    Returns: List of (tile_size, gemm1_tactic, gemm2_tactic)
    """
    tactics = []
    # tile_size=256 (2-CTA) is enabled: the gemm1(2-CTA)/gemm2(1-CTA) layout
    # mismatch that caused incorrect results (#3067) is fixed by parameterizing
    # get_blackwell_gemm2_valid_tactics on tile_size (#3171). Mirrors main's
    # get_moe_valid_tactics over VALID_TILE_SIZES.
    for tile_size in VALID_TILE_SIZES:
        gemm1_tactics = get_blackwell_gemm1_valid_tactics(tile_size)
        gemm2_tactics = get_blackwell_gemm2_valid_tactics(tile_size)
        for gemm1_tactic, gemm2_tactic in itertools.product(
            gemm1_tactics, gemm2_tactics
        ):
            tactics.append((tile_size, gemm1_tactic, gemm2_tactic))
    return tactics


# Canonical list of tile_sizes the autotuner is allowed to pick.  Used by
# ``CuteDslMoEWrapper`` to size its preallocated kernel-output buffers so
# every tactic in the arch-specific tactic lists can reuse the prealloc,
# regardless of which tile_size the autotuner picks at runtime.  Adding a
# new tile_size here automatically widens the prealloc.
VALID_TILE_SIZES: Tuple[int, ...] = (128, 256)


# =============================================================================
# Rubin (SM107) Tactics
# =============================================================================
# Rubin tactics use 3-tuple mma_tiler/mma_inst_shape and support B-reuse.
# Fixed K dimensions for FP4: mma_tiler_k=256, mma_inst_k=128
#
# Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
# where mma_tiler = (M, N, K) and mma_inst_shape = (M', N, K')
def get_rubin_gemm1_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Rubin tactics for GEMM1 (Gather + SwiGLU Fusion).

    Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_k = 256
    mma_inst_k = 128

    # (mma_tiler_m, mma_inst_m) candidates — B-reuse when tiler_m = 2 * inst_m
    mma_m_candidates = [
        (128, 128),  # no B-reuse, 1CTA
        (256, 256),  # no B-reuse, 2CTA
        (256, 128),  # B-reuse, 1CTA
        (512, 256),  # B-reuse, 2CTA
    ]
    mma_n_candidates = [128, 256]
    cluster_shape_mn_candidates = [(1, 1), (2, 1)]
    raster_along_m_candidates = [False]

    valid_tactics = []
    for (
        mma_tiler_m,
        mma_inst_m,
    ), mma_n, cluster_shape_mn, raster_along_m in itertools.product(
        mma_m_candidates,
        mma_n_candidates,
        cluster_shape_mn_candidates,
        raster_along_m_candidates,
    ):
        if mma_tiler_m > tile_size:
            continue
        # GEMM1 is a gather GEMM: mma_tiler_m must equal tile_size so that
        # each CTA's tile exactly covers one moe_sort tile in the M dimension.
        # Smaller mma_tiler_m causes incorrect gather indexing.
        if mma_tiler_m != tile_size:
            continue
        if cluster_shape_mn[0] * mma_tiler_m > tile_size:
            continue
        # 2CTA (mma_inst_m=256) requires even cluster_shape_m
        if mma_inst_m == 256 and cluster_shape_mn[0] % 2 != 0:
            continue

        mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
        mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)
        valid_tactics.append(
            (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
        )

    return valid_tactics


def get_rubin_gemm2_valid_tactics(tile_size: int) -> List[Tuple]:
    """Get valid Rubin tactics for GEMM2 (Finalize Fusion).

    Format: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
    """
    mma_tiler_k = 256
    mma_inst_k = 128

    mma_m_candidates = [
        (128, 128),
        (256, 256),
        (256, 128),
        (512, 256),
    ]
    mma_n_candidates = [128, 256]
    # Restrict to cluster_shape_n=1 only. The Rubin finalize kernel
    # triggers illegal memory accesses with cluster_shape_n>1 at
    # larger token counts (non-deterministic, routing-dependent).
    cluster_shape_mn_candidates = [(1, 1), (2, 1)]
    raster_along_m_candidates = [False]

    valid_tactics = []
    for (
        mma_tiler_m,
        mma_inst_m,
    ), mma_n, cluster_shape_mn, raster_along_m in itertools.product(
        mma_m_candidates,
        mma_n_candidates,
        cluster_shape_mn_candidates,
        raster_along_m_candidates,
    ):
        # tile_idx_to_expert_idx has one entry per routing tile (tile_size
        # rows). mma_tiler_m must equal tile_size so that the CTA tile
        # aligns with the routing tile — matching TRT-LLM's enforcement.
        if mma_tiler_m != tile_size:
            continue
        if cluster_shape_mn[0] * mma_tiler_m > tile_size:
            continue
        # 2CTA (mma_inst_m=256) requires even cluster_shape_m
        if mma_inst_m == 256 and cluster_shape_mn[0] % 2 != 0:
            continue

        mma_tiler = (mma_tiler_m, mma_n, mma_tiler_k)
        mma_inst_shape = (mma_inst_m, mma_n, mma_inst_k)
        valid_tactics.append(
            (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m)
        )

    return valid_tactics


def get_rubin_moe_valid_tactics() -> List[Tuple]:
    """Get all valid Rubin MoE tactic combinations.

    Returns: List of (tile_size, gemm1_tactic, gemm2_tactic)
    """
    tactics = []
    # Only tile_size=128 is enabled. tile_size=256 with B-reuse causes
    # illegal memory accesses for certain GEMM2 tactic configurations and
    # is disabled until the kernel bug is fixed (mirrors the Blackwell
    # restriction in get_blackwell_moe_valid_tactics).
    for tile_size in [128]:
        gemm1_tactics = get_rubin_gemm1_valid_tactics(tile_size)
        gemm2_tactics = get_rubin_gemm2_valid_tactics(tile_size)
        for gemm1_tactic, gemm2_tactic in itertools.product(
            gemm1_tactics, gemm2_tactics
        ):
            tactics.append((tile_size, gemm1_tactic, gemm2_tactic))
    return tactics


# =============================================================================
# Pre-generated tactic sets
# =============================================================================

ALL_BLACKWELL_MOE_TACTICS = get_blackwell_moe_valid_tactics()
ALL_RUBIN_MOE_TACTICS = get_rubin_moe_valid_tactics()


DEFAULT_BLACKWELL_MOE_TACTIC = (
    128,
    ((128, 128), (1, 1), False),
    ((128, 128), (1, 1), False),
)

DEFAULT_RUBIN_MOE_TACTIC = (
    128,
    ((128, 128, 256), (128, 128, 128), (1, 1), False),
    ((128, 128, 256), (128, 128, 128), (1, 1), False),
)

# =============================================================================
# Tactic parameter extraction
# =============================================================================


def _is_rubin_tactic(tactic: Tuple) -> bool:
    """Detect whether a tactic is Rubin format by checking sub-tactic length.

    Blackwell sub-tactic: (mma_tiler_mn, cluster_shape_mn, raster_along_m) — 3 elements
    Rubin sub-tactic: (mma_tiler, mma_inst_shape, cluster_shape_mn, raster_along_m) — 4 elements
    """
    _, gemm1_tactic, _ = tactic
    return len(gemm1_tactic) == 4


def _extract_tactic_params(tactic: Tuple) -> Dict[str, Any]:
    """Extract parameters from a MoE tactic tuple.

    Handles both Blackwell and Rubin formats transparently.

    Returns:
        Dictionary with all tactic parameters. For Rubin tactics, includes
        'gemm1_mma_tiler', 'gemm1_mma_inst_shape', etc. in addition to
        the standard keys.
    """
    tile_size, gemm1_tactic, gemm2_tactic = tactic

    if _is_rubin_tactic(tactic):
        (
            gemm1_mma_tiler,
            gemm1_mma_inst_shape,
            gemm1_cluster_shape_mn,
            gemm1_raster_along_m,
        ) = gemm1_tactic
        (
            gemm2_mma_tiler,
            gemm2_mma_inst_shape,
            gemm2_cluster_shape_mn,
            gemm2_raster_along_m,
        ) = gemm2_tactic
        return {
            "tile_size": tile_size,
            "is_rubin": True,
            "gemm1_mma_tiler_mn": (gemm1_mma_tiler[0], gemm1_mma_tiler[1]),
            "gemm1_cluster_shape_mn": gemm1_cluster_shape_mn,
            "gemm1_raster_along_m": gemm1_raster_along_m,
            "gemm1_mma_tiler": gemm1_mma_tiler,
            "gemm1_mma_inst_shape": gemm1_mma_inst_shape,
            "gemm2_mma_tiler_mn": (gemm2_mma_tiler[0], gemm2_mma_tiler[1]),
            "gemm2_cluster_shape_mn": gemm2_cluster_shape_mn,
            "gemm2_raster_along_m": gemm2_raster_along_m,
            "gemm2_mma_tiler": gemm2_mma_tiler,
            "gemm2_mma_inst_shape": gemm2_mma_inst_shape,
        }
    else:
        gemm1_mma_tiler_mn, gemm1_cluster_shape_mn, gemm1_raster_along_m = gemm1_tactic
        gemm2_mma_tiler_mn, gemm2_cluster_shape_mn, gemm2_raster_along_m = gemm2_tactic
        return {
            "tile_size": tile_size,
            "is_rubin": False,
            "gemm1_mma_tiler_mn": gemm1_mma_tiler_mn,
            "gemm1_cluster_shape_mn": gemm1_cluster_shape_mn,
            "gemm1_raster_along_m": gemm1_raster_along_m,
            "gemm1_mma_tiler": None,
            "gemm1_mma_inst_shape": None,
            "gemm2_mma_tiler_mn": gemm2_mma_tiler_mn,
            "gemm2_cluster_shape_mn": gemm2_cluster_shape_mn,
            "gemm2_raster_along_m": gemm2_raster_along_m,
            "gemm2_mma_tiler": None,
            "gemm2_mma_inst_shape": None,
        }


def _get_arch_tactics() -> List[Tuple]:
    """Return the tactic list appropriate for the current GPU architecture."""
    if not torch.cuda.is_available():
        return ALL_BLACKWELL_MOE_TACTICS
    major, minor = get_compute_capability(torch.device("cuda"))
    if major == 10 and minor == 7:
        return ALL_RUBIN_MOE_TACTICS
    return ALL_BLACKWELL_MOE_TACTICS


def _get_default_tactic() -> Tuple:
    """Return the default tactic for the current GPU architecture."""
    if not torch.cuda.is_available():
        return DEFAULT_BLACKWELL_MOE_TACTIC
    major, minor = get_compute_capability(torch.device("cuda"))
    if major == 10 and minor == 7:
        return DEFAULT_RUBIN_MOE_TACTIC
    return DEFAULT_BLACKWELL_MOE_TACTIC


# =============================================================================
# TunableRunner
# =============================================================================


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    """TunableRunner for CuteDSL NVFP4 MoE kernels.

    Supports both Blackwell (SM100) and Rubin (SM107) architectures.
    Tactic format is architecture-dependent — see _extract_tactic_params.
    """

    def __init__(
        self,
        forward_impl: Callable,
        num_experts: int,
        top_k: int,
        num_local_experts: int,
        local_expert_offset: int = 0,
        use_fused_finalize: bool = True,
        output_dtype: torch.dtype = torch.bfloat16,
        enable_pdl: bool = True,
    ):
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.use_fused_finalize = use_fused_finalize
        self.output_dtype = output_dtype
        self.enable_pdl = enable_pdl

        # Helper that builds a deterministic balanced approx-max-load
        # assignment for token_selected_experts during autotune profiling.
        # See _inputs_helper.py for rationale -- the random tensor_initializer
        # for input #2 produces non-deterministic and unrealistic per-expert
        # load distributions, biasing autotune picks at marginal cells.
        self._inputs_helper = CuteDslMoEInputsHelper(
            num_experts, top_k, num_local_experts, local_expert_offset
        )

        # Instance-level so dummy expert IDs span all local experts
        # (randint(0, num_experts)) for realistic profiling.
        self.tuning_config = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    input_idx=(0, 1, 2, 3, 11),
                    dim_idx=(0, 0, 0, 0, 0),
                    # Bare callables: autotuner adapts the bucket set to
                    # the actual input dim (matches the
                    # _FP8_GEMM_SM100_TUNING_CONFIG pattern in
                    # `gemm/gemm_base.py`).
                    gen_tuning_buckets=get_hybrid_num_tokens_buckets,
                    map_to_tuning_buckets=map_to_hybrid_bucket_uncapped,
                    tensor_initializers=[
                        # 0: x — FP4 quantized input (uint8 packed). Seeded
                        # for cross-process determinism of autotune picks
                        # (matches trt-llm's seed=515 convention).
                        lambda shapes, dtype, device: torch.randint(
                            0,
                            256,
                            shapes,
                            dtype=torch.uint8,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(515),
                        ),
                        # 1: x_sf — FP8 scale factors (uint8). Seeded.
                        lambda shapes, dtype, device: torch.randint(
                            1,
                            128,
                            shapes,
                            dtype=torch.uint8,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(515),
                        ),
                        # 2: token_selected_experts — output is overwritten
                        # by inputs_pre_hook (CuteDslMoEInputsHelper), but
                        # seed the initializer too in case the hook is ever
                        # disabled.
                        lambda shapes, dtype, device: torch.randint(
                            0,
                            max(num_experts, 1),
                            shapes,
                            dtype=torch.int32,
                            device=device,
                            generator=torch.Generator(device=device).manual_seed(515),
                        ),
                        # 3: token_final_scales — softmax-normalized. Seeded.
                        lambda shapes, dtype, device: torch.softmax(
                            torch.randn(
                                shapes,
                                device=device,
                                generator=torch.Generator(device=device).manual_seed(
                                    515
                                ),
                            ),
                            dim=-1,
                        ).to(torch.float32),
                        # 11: moe_output — output buffer
                        lambda shapes, dtype, device: torch.empty(
                            shapes, dtype=dtype, device=device
                        ),
                    ],
                ),
            ),
            inputs_pre_hook=self._inputs_helper.inputs_pre_hook,
            # Cold-L2 measurement matches TRT-LLM's
            # CuteDslFusedMoENvfp4Runner.tuning_config; flushing L2
            # between profile iterations yields autotune timings
            # representative of production cold-cache conditions.
            use_cold_l2_cache=True,
        )

    def __hash__(self):
        return hash(
            (
                self.num_experts,
                self.top_k,
                self.num_local_experts,
                self.local_expert_offset,
                self.use_fused_finalize,
                self.output_dtype,
            )
        )

    def get_valid_tactics(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
    ) -> List[Tuple[Any, ...]]:
        """Return valid tactics filtered by can_implement checks.

        Validates each candidate tactic against both GEMM1 and GEMM2 kernel
        can_implement methods using the actual problem dimensions from inputs.
        Supports both Blackwell and Rubin architectures.
        """
        import cutlass
        from .moe_utils import get_max_num_permuted_tokens

        x = inputs[0]
        w1_weight = inputs[4]

        num_tokens = x.shape[0]
        hidden_size = x.shape[1] * 2  # FP4 packed
        num_local_experts = w1_weight.shape[0]
        intermediate_size = w1_weight.shape[1] // 2  # gate+up fused

        ab_dtype = cutlass.Float4E2M1FN
        sf_dtype = cutlass.Float8E4M3FN
        sf_vec_size = 16
        gemm1_c_dtype = cutlass.Float4E2M1FN
        gemm2_out_dtype = cutlass.BFloat16

        all_tactics = _get_arch_tactics()
        valid_tactics = []

        for tactic in all_tactics:
            tile_size, gemm1_tactic, gemm2_tactic = tactic
            permuted_m = get_max_num_permuted_tokens(
                num_tokens, self.top_k, self.num_local_experts, tile_size
            )

            if _is_rubin_tactic(tactic):
                from .rubin import (
                    Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel,
                    Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
                )

                gemm1_mma_tiler, gemm1_mma_inst_shape, gemm1_cluster_shape_mn, _ = (
                    gemm1_tactic
                )
                gemm2_mma_tiler, gemm2_mma_inst_shape, gemm2_cluster_shape_mn, _ = (
                    gemm2_tactic
                )

                gemm1_ok = Sm107BlockScaledContiguousGatherGroupedGemmSwigluFusionKernel.can_implement(
                    a_dtype=ab_dtype,
                    b_dtype=ab_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm1_c_dtype,
                    mma_inst_shape=gemm1_mma_inst_shape,
                    mma_tiler=gemm1_mma_tiler,
                    cluster_shape_mn=gemm1_cluster_shape_mn,
                    m=permuted_m,
                    n=2 * intermediate_size,
                    k=hidden_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
                gemm2_ok = Sm107BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    a_dtype=ab_dtype,
                    b_dtype=ab_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm2_out_dtype,
                    mma_inst_shape=gemm2_mma_inst_shape,
                    mma_tiler=gemm2_mma_tiler,
                    cluster_shape_mn=gemm2_cluster_shape_mn,
                    m=permuted_m,
                    n=hidden_size,
                    k=intermediate_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
            else:
                from .blackwell import (
                    BlockScaledContiguousGatherGroupedGemmKernel,
                    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
                )

                gemm1_mma_tiler_mn, gemm1_cluster_shape_mn, _ = gemm1_tactic
                gemm2_mma_tiler_mn, gemm2_cluster_shape_mn, _ = gemm2_tactic

                gemm1_ok = BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
                    ab_dtype=ab_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    c_dtype=gemm1_c_dtype,
                    mma_tiler_mn=gemm1_mma_tiler_mn,
                    cluster_shape_mn=gemm1_cluster_shape_mn,
                    m=permuted_m,
                    n=2 * intermediate_size,
                    k=hidden_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    c_major="n",
                )
                gemm2_ok = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    ab_dtype=ab_dtype,
                    sf_dtype=sf_dtype,
                    sf_vec_size=sf_vec_size,
                    out_dtype=gemm2_out_dtype,
                    mma_tiler_mn=gemm2_mma_tiler_mn,
                    cluster_shape_mn=gemm2_cluster_shape_mn,
                    m=permuted_m,
                    n=hidden_size,
                    k=intermediate_size,
                    l=num_local_experts,
                    a_major="k",
                    b_major="k",
                    out_major="n",
                )

            if gemm1_ok and gemm2_ok:
                valid_tactics.append(tactic)

        if not valid_tactics:
            logger.warning(
                "No valid tactics found for problem dims "
                "(tokens=%d, hidden=%d, intermediate=%d, experts=%d, top_k=%d). "
                "Falling back to default tactic.",
                num_tokens,
                hidden_size,
                intermediate_size,
                num_local_experts,
                self.top_k,
            )
            valid_tactics = [_get_default_tactic()]

        return valid_tactics

    def forward(  # type: ignore[override]
        self,
        inputs: List[torch.Tensor],
        tactic: Tuple[Any, ...] = None,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the MoE forward pass with the specified tactic."""
        if tactic is None or tactic == -1:
            tactic = _get_default_tactic()

        params = _extract_tactic_params(tactic)

        (
            x,
            x_sf,
            token_selected_experts,
            token_final_scales,
            w1_weight,
            w1_weight_sf,
            w1_alpha,
            fc2_input_scale,
            w2_weight,
            w2_weight_sf,
            w2_alpha,
            *optional_inputs,
        ) = inputs

        moe_output = optional_inputs[0] if optional_inputs else None

        return self.forward_impl(
            x=x,
            x_sf=x_sf,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w1_alpha=w1_alpha,
            fc2_input_scale=fc2_input_scale,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
            w2_alpha=w2_alpha,
            num_experts=self.num_experts,
            top_k=self.top_k,
            num_local_experts=self.num_local_experts,
            local_expert_offset=self.local_expert_offset,
            tile_size=params["tile_size"],
            gemm1_mma_tiler_mn=params["gemm1_mma_tiler_mn"],
            gemm1_cluster_shape_mn=params["gemm1_cluster_shape_mn"],
            gemm2_mma_tiler_mn=params["gemm2_mma_tiler_mn"],
            gemm2_cluster_shape_mn=params["gemm2_cluster_shape_mn"],
            gemm1_mma_tiler=params["gemm1_mma_tiler"],
            gemm1_mma_inst_shape=params["gemm1_mma_inst_shape"],
            gemm2_mma_tiler=params["gemm2_mma_tiler"],
            gemm2_mma_inst_shape=params["gemm2_mma_inst_shape"],
            output_dtype=self.output_dtype,
            use_fused_finalize=self.use_fused_finalize,
            moe_output=moe_output,
            enable_pdl=self.enable_pdl,
            **kwargs,
        )


# =============================================================================
# Utility Functions
# =============================================================================


def print_all_tactics():
    """Print all valid MoE tactics for debugging."""
    for label, tactics in [
        ("Blackwell", ALL_BLACKWELL_MOE_TACTICS),
        ("Rubin", ALL_RUBIN_MOE_TACTICS),
    ]:
        logger.info("%s MoE tactics: %d", label, len(tactics))
        for i, tactic in enumerate(tactics):
            tile_size, gemm1_tactic, gemm2_tactic = tactic
            logger.info(
                "  Tactic %d: tile_size=%s, gemm1=%s, gemm2=%s",
                i,
                tile_size,
                gemm1_tactic,
                gemm2_tactic,
            )
