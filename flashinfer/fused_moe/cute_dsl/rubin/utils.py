# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Rubin (SM107) specific kernel utilities.

Re-exports shared utilities from common/kernel_utils.py and adds
Rubin-specific fmin (without explicit return type argument).
"""

from typing import Union

import cutlass
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op

# Re-export all shared utilities so existing imports continue to work
from ..common.kernel_utils import (  # noqa: F401
    _Pointer,
    atomic_add_func,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    is_power_of_2,
    make_ptr,
    sigmoid_f32,
    silu_f32,
    vectorized_atomic_add_bf16x8,
    vectorized_atomic_add_fp32x2,
)


# ============================================================================
# Rubin-specific functions
# ============================================================================


@dsl_user_op
def fmin(
    a: Union[float, cutlass.Float32],
    b: Union[float, cutlass.Float32],
    *,
    nan=False,
    loc=None,
    ip=None,
) -> cutlass.Float32:
    return cutlass.Float32(
        nvvm.fmin(
            cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            nan=nan,
            loc=loc,
            ip=ip,
        )
    )
