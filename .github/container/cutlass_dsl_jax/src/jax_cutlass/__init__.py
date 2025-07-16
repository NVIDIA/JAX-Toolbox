# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .primitive import cutlass_call
from .types import jax_to_cutlass_dtype, from_dlpack, JaxArray
from .compile import release_compile_cache, initialize_cutlass_dsl
from .version import __version__, __version_info__

# This explicit init method ensures that we avoid initialization at
# unexpected times. TODO: try to remove the need for this initialization.
initialize_cutlass_dsl()

__all__ = [
    "cutlass_call",
    "jax_to_cutlass_dtype",
    "from_dlpack",
    "JaxArray",
    "release_compile_cache",
    "__version__",
    "__version_info__",
]
