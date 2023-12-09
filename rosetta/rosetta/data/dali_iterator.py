# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import time

from nvidia.dali.plugin.jax.clu import peekable_data_iterator

from jax.experimental import multihost_utils

import jax


def get_dali_dataset(
    config,
    ds_shard_id,
    ds_num_shards,
    feature_converter
):  
    assert not bool(feature_converter), 'Passing `feature_converter_cls` is not supported'
    
    # TODO(awolant): Implement support for multiple hosts
    assert ds_shard_id == 0, 'DALI does not support multi host training'
    assert ds_num_shards == 1, 'DALI does not support multi host training'
    
    # For multihost training we have ds_num_shards > 1 and ds_shard_id points to the shard id of the current host.
    # For single host training we have ds_num_shards == 1 and ds_shard_id == 0.
    
    seed = config.seed
    if seed is None:
        # Use a shared timestamp across devices as the seed.
        seed = int(multihost_utils.broadcast_one_to_all(np.int32(time.time())))
    config.seed = seed
    
    print("Local devices: ", jax.local_devices())
    
    
    pass