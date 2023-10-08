# Copyright (c) 2022-2023 NVIDIA Corporation
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

import numpy as np
from typing import List

def pow2list(n: int):
    pow2 = []
    i = 1
    while i < n:
        pow2.append(i)
        i *= 2

    pow2.append(n)
    return pow2

def triton_textencode(text_batch: List[str]):
    enc = np.array([[np.char.encode(i, 'utf-8')] for i in text_batch])
    enc = np.reshape(enc, (enc.shape[0], 1))

    return enc

