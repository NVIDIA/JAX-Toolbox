#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def prime_factors(n: int):
    """Return the prime factors of n as a list (ascending).
    For n < 1, raises ValueError. For 1, returns [1].
    """
    if n < 1:
        raise ValueError("0 has infinitely many divisors; factorization undefined.")
    if n == 1:
        return (1,)

    factors = []

    # Pull out factors of 2
    while n % 2 == 0:
        factors.append(2)
        n //= 2

    # Pull out factors of 3
    while n % 3 == 0:
        factors.append(3)
        n //= 3

    # Check 6k ± 1 candidates
    f = 5
    while f * f <= n:
        for cand in (f, f + 2):  # 6k-1 and 6k+1
            while n % cand == 0:
                factors.append(cand)
                n //= cand
        f += 6

    # If what's left is >1, it's prime
    if n > 1:
        factors.append(n)

    return tuple(sorted(factors))
