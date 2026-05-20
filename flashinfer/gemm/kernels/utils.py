"""
Copyright (c) 2024 by FlashInfer team.

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

from ...fused_moe.utils import last_positive_power_of_2

_SM100_MMA_TILER_MN_CANDIDATES = [
    (128, 64),
    (256, 64),
    (128, 128),
    (256, 128),
    (128, 192),
    (256, 192),
    (128, 256),
    (256, 256),
]

# Tactic cache: (n, real_k, sm_count) -> dict[(m_bucket, is_8_aligned) -> tactic_tuple]
# Bounded by the number of unique (N, K) pairs in the model (typically < 50).
_SM100_MM_FP4_TACTIC_CACHE: dict[tuple, dict] = {}

# M bucket boundaries — powers of 2 for fast bucketing via
# last_positive_power_of_2 (imported from flashinfer.fused_moe.utils).
# Each bucket is precomputed for both 8-aligned and non-8-aligned M,
# keyed as (bucket, is_8_aligned).
_M_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def _compute_tactic_for_m(rep_m, n, real_k, sm_count, m_aligned):
    """Compute the best tactic for a specific (M, N, K) on a GPU with sm_count SMs.

    Selects swap_ab, tile shape, and cluster shape sequentially:

    1. **swap_ab**: Swap A and B operands when M is small (8-16) and
       8-aligned, putting the larger N dimension on the M-axis to increase
       the number of CTAs.  Also swaps when N is not 8-aligned (required
       for memory alignment).

    2. **Tile shape**: Scores all 8 candidates from _SM100_MMA_TILER_MN_CANDIDATES.
       The score balances three factors:
         - Tile quantization: M and N padding waste from rounding up to tile
           boundaries. Smaller tiles waste less for small dimensions.
         - Wave quantization: How well total CTAs fill the available SMs.
           Ideal when total_ctas is a multiple of sm_count.
         - Tile throughput: Larger, balanced tiles have higher per-CTA
           throughput. Penalized for small K (<=2048) where the K-pipeline
           can't hide launch latency of large tiles.
       The combined score is: m_eff * wave_eff * n_eff * tile_throughput.

    3. **Cluster shape**: For narrow GEMMs (prob_m fits in one tile row),
       clusters in the N dimension (up to 4) for scale-factor multicast.
       Otherwise uses (1, 1).  tile_m=256 forces cluster_m=2 (HW constraint).

    4. **Prefetch**: Disabled.
    """
    n_aligned = n % 8 == 0

    swap_ab = False
    if m_aligned and 8 <= rep_m <= 16 and n > rep_m:
        swap_ab = True
    if not swap_ab and not n_aligned and m_aligned:
        swap_ab = True

    prob_m = n if swap_ab else rep_m
    prob_n = rep_m if swap_ab else n

    # Small-K penalty factor (loop-invariant).
    if real_k <= 1024:
        large_tile_penalty = 0.50
    elif real_k <= 2048:
        large_tile_penalty = 0.80
    else:
        large_tile_penalty = 1.0

    # Score all 8 tile candidates.
    max_tile_area = 256 * 256
    best_tile_m = 128
    best_tile_n = 128
    best_score = -1.0

    for tile_m, tile_n in _SM100_MMA_TILER_MN_CANDIDATES:
        n_tiles = (prob_n + tile_n - 1) // tile_n
        n_eff = prob_n / (n_tiles * tile_n)
        tile_area_factor = ((tile_m * tile_n) / max_tile_area) ** 0.5
        tile_bal = min(tile_m, tile_n) / max(tile_m, tile_n)
        tile_tp = tile_area_factor * (tile_bal**0.25)
        ns = n_eff * tile_tp

        if tile_m * tile_n > 128 * 128:
            ns *= large_tile_penalty

        m_tiles = (prob_m + tile_m - 1) // tile_m
        total_ctas = m_tiles * n_tiles
        num_waves = (total_ctas + sm_count - 1) // sm_count
        score = prob_m * total_ctas * ns / (tile_m * num_waves * sm_count)
        if score > best_score:
            best_score = score
            best_tile_m = tile_m
            best_tile_n = tile_n

    # Cluster: N-only for small prob_m, else (1,1).
    tiles_on_n = (prob_n + best_tile_n - 1) // best_tile_n
    if prob_m <= best_tile_m:
        if tiles_on_n % 4 == 0 or tiles_on_n > 10:
            cga_n = 4
        elif tiles_on_n % 2 == 0:
            cga_n = 2
        else:
            cga_n = 1
        cga_m = 1
    else:
        cga_m = 1
        cga_n = 1

    if best_tile_m == 256 and cga_m < 2:
        cga_m = 2

    return (
        (best_tile_m, best_tile_n),
        (cga_m, cga_n),
        swap_ab,
        False,
        "sm100",
        None,
    )


def _select_sm100_mm_fp4_cute_dsl_tactic(m, n, real_k, sm_count):
    """Select the best tactic for mm_fp4(backend='cute-dsl').

    On the first call for a given (N, K), precomputes the optimal tactic
    for each M bucket (~13 buckets, ~55-86 usec).  Subsequent calls with
    any M just look up the bucket — runs in ~0.2 usec.

    Args:
        m: M dimension of the GEMM problem.
        n: N dimension of the GEMM problem.
        real_k: K dimension (unpacked, i.e. 2x the packed FP4 dimension).
        sm_count: Number of SMs on the target GPU.

    Returns:
        Tactic tuple: (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch,
                        kernel_type, use_tma_store)
    """
    cache_key = (n, real_k, sm_count)
    bucket_tactics = _SM100_MM_FP4_TACTIC_CACHE.get(cache_key)
    if bucket_tactics is None:
        bucket_tactics = {}
        for rep_m in _M_BUCKETS:
            for aligned in (True, False):
                bucket_tactics[(rep_m, aligned)] = _compute_tactic_for_m(
                    rep_m, n, real_k, sm_count, aligned
                )
        _SM100_MM_FP4_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(last_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[(bucket, m % 8 == 0)]


# Tactic cache for bmm_fp8 cute-dsl.  Keyed by problem signature; each entry
# maps an M bucket to an integer index into the arch-specific AUTOTUNE_CONFIGS.
# Includes the dtype/layout/batch tuple in the key because they affect config
# validity (different problem signatures expose different valid index sets).
_BMM_FP8_TACTIC_CACHE: dict[tuple, dict] = {}


def _score_bmm_fp8_tile(m, n, k, batch, sm_count, tile_m, tile_n, large_tile_penalty):
    """Score a (tile_m, tile_n) candidate for a bmm_fp8 problem.

    Score = m_eff * wave_eff * n_eff * tile_throughput, where each factor
    is in (0, 1] (except tile_throughput which scales with tile area).
    This matches the docstring intent of the mm_fp4 heuristic but drops a
    spurious `m_tiles` factor that appeared in the original formula's
    return expression (`m * total_ctas / (tile_m * ...)` expands to
    `m_tiles * m_eff * wave_eff * ...`).  That extra factor systematically
    rewarded smaller tile_m and dominated the tile-area term, which is why
    the FP4 score formula picked (64, 128) for FP8 problems where larger
    2-CTA tiles are measurably faster on B200.

    Factors:
      - m_eff = m / (m_tiles * tile_m): M tile quantization (1 = no waste).
      - n_eff: same for N.
      - wave_eff = total_ctas / (num_waves * sm_count): how full each wave
        of CTAs is.  Batched problems multiply total_ctas by batch.
      - tile_tp = sqrt(area / max_area) * tile_bal^0.25: rewards large,
        balanced tiles; large_tile_penalty kicks in when K is small enough
        that big tiles can't hide pipeline latency.
    """
    m_tiles = (m + tile_m - 1) // tile_m
    n_tiles = (n + tile_n - 1) // tile_n
    m_eff = m / (m_tiles * tile_m)
    n_eff = n / (n_tiles * tile_n)

    max_tile_area = 256 * 256
    # Exponent 0.75 (vs 0.5 for mm_fp4): empirically tuned for B200 FP8 to
    # better counterweight the implicit `m_tiles` reward in wave_eff so that
    # larger tiles win on prefill / deep-K shapes where they are measurably
    # faster.  0.5 left wave_eff dominating; 1.0 over-biases toward
    # unrealistically large tiles on small problems.
    tile_area_factor = ((tile_m * tile_n) / max_tile_area) ** 0.75
    tile_bal = min(tile_m, tile_n) / max(tile_m, tile_n)
    tile_tp = tile_area_factor * (tile_bal**0.25)

    if tile_m * tile_n > 128 * 128:
        tile_tp *= large_tile_penalty

    total_ctas = batch * m_tiles * n_tiles
    num_waves = (total_ctas + sm_count - 1) // sm_count
    wave_eff = total_ctas / (num_waves * sm_count)

    return m_eff * wave_eff * n_eff * tile_tp


def _extract_config_min_m_and_tile(config):
    """Return (min_m, tile_m, tile_n) for a SM100_AUTOTUNE_CONFIGS entry.

    `min_m = cta_tile_m` is the smallest M that satisfies the hard
    per-CTA tile constraint.  _can_implement_config_sm100 is more
    conservative (requires `m >= cta_tile_m * cluster_m`), but empirically
    the kernel runs correctly on B200 FP8 even when the cluster is
    under-filled in the M direction (e.g., tactic 0's cluster=(2,1)
    happily handles m=64 with correct output).  Adopting the looser
    constraint here lets the heuristic consider tactic 0 for batched
    small-m shapes where it is measurably the fastest option.  All other
    validity (dtype, alignment, layout, n, k, batch) is m-independent and
    handled by the one get_valid_sm100_configs(max_m, ...) sweep.
    """
    # SM100 config: (mma_tiler_mn, cluster_shape_mn, use_2cta_instrs, ...)
    mma_tiler_mn = config[0]
    use_2cta_instrs = config[2]
    cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
    tile_m, tile_n = mma_tiler_mn[0], mma_tiler_mn[1]
    return cta_tile_m, tile_m, tile_n


def _build_bmm_fp8_bucket_tactics(
    n, k, batch, ab_dtype, c_dtype, a_major, b_major, c_major, sm_count
):
    """Build the {m_bucket: config_index} dict for one problem signature.

    Performs one expensive validity sweep at max-M (filters m-independent
    constraints: dtype, alignment, layout, n, k, batch).  For each smaller
    M bucket, the m-dependent pre-check (`m >= cta_tile_m * cluster_m`) is
    applied with cheap Python arithmetic on the cached config metadata.

    This trades 13×13 expensive `can_implement` calls for 13 expensive
    calls + 13×13 integer comparisons, keeping first-call cost bounded
    even when the heuristic fires on a fresh problem signature.

    Falls back to config index 0 (the documented "default fallback
    config") when no valid config exists for a bucket.
    """
    from .bmm_fp8_wrapper import (
        SM100_AUTOTUNE_CONFIGS as CONFIGS,
        get_valid_sm100_configs as get_valid,
    )

    # One expensive validity sweep at max-M filters every m-independent
    # constraint at once; smaller buckets only need the cheap m-pre-check.
    max_valid_indices = get_valid(
        _M_BUCKETS[-1], n, k, batch, ab_dtype, c_dtype, a_major, b_major, c_major
    )
    if not max_valid_indices:
        return {rep_m: 0 for rep_m in _M_BUCKETS}

    # Cache per-config min-m + tile dims so the inner loop is pure Python arithmetic.
    config_info = {
        idx: _extract_config_min_m_and_tile(CONFIGS[idx])
        for idx in max_valid_indices
    }

    if k <= 1024:
        large_tile_penalty = 0.50
    elif k <= 2048:
        large_tile_penalty = 0.80
    else:
        large_tile_penalty = 1.0

    bucket_tactics: dict = {}
    for rep_m in _M_BUCKETS:
        best_idx = 0
        best_score = -1.0
        any_valid = False
        for idx in max_valid_indices:
            min_m, tile_m, tile_n = config_info[idx]
            if rep_m < min_m:
                continue
            score = _score_bmm_fp8_tile(
                rep_m, n, k, batch, sm_count, tile_m, tile_n, large_tile_penalty
            )
            if not any_valid or score > best_score:
                any_valid = True
                best_score = score
                best_idx = idx
        bucket_tactics[rep_m] = best_idx
    return bucket_tactics


def _select_sm100_bmm_fp8_cute_dsl_tactic(
    m, n, k, batch, ab_dtype, c_dtype, a_major, b_major, c_major, sm_count
):
    """Select the best config index for bmm_fp8(backend='cute-dsl') on SM100.

    Mirrors _select_sm100_mm_fp4_cute_dsl_tactic but scores discrete
    SM100_AUTOTUNE_CONFIGS entries (filtered to those that can implement
    the problem) and returns an integer index instead of building a tactic
    tuple from scratch.

    On the first call for a given problem signature, precomputes the best
    index for each M bucket and caches it; subsequent calls with any M
    just look up the bucket.

    Args:
        m, n, k: Per-batch GEMM dimensions.
        batch: Batch size.
        ab_dtype, c_dtype: cutlass.Numeric subclasses for A/B and C.
        a_major, b_major, c_major: Major-dim labels ("m"/"k", "k"/"n", "n").
        sm_count: Number of SMs on the target GPU.

    Returns:
        Config index (int) into SM100_AUTOTUNE_CONFIGS.  Falls back to 0
        when no valid config exists for the bucket (matches the runner's
        previous fallback behavior).
    """
    cache_key = (
        n, k, batch, ab_dtype, c_dtype, a_major, b_major, c_major, sm_count,
    )
    bucket_tactics = _BMM_FP8_TACTIC_CACHE.get(cache_key)
    if bucket_tactics is None:
        bucket_tactics = _build_bmm_fp8_bucket_tactics(
            n, k, batch, ab_dtype, c_dtype, a_major, b_major, c_major, sm_count
        )
        _BMM_FP8_TACTIC_CACHE[cache_key] = bucket_tactics

    bucket = min(last_positive_power_of_2(m), _M_BUCKETS[-1])
    return bucket_tactics[bucket]
