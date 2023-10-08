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
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from multiprocessing import shared_memory
import logging
import time

@dataclass
class SharedNPMeta:
    shmem_name: str
    shape: Tuple[int]
    dtype: np.dtype

class SharedNPArray:
    metadata: SharedNPMeta
    shmem: shared_memory.SharedMemory
    array: np.ndarray
    closed: bool = False
    name: Optional[str] = None

    def __init__(self, arr_to_share: Optional[np.ndarray]=None, metadata: Optional[SharedNPMeta]=None, name: Optional[str]=None):
        if arr_to_share is not None:
            start_time = time.time()
            # Creates shared memory array
            assert metadata is None, "Please provide either an array_to_share or metadata"

            nbytes = arr_to_share.nbytes
            self.shmem = shared_memory.SharedMemory(create=True, size=arr_to_share.nbytes)
            # logging.warning(f'creating {self.shmem.name} with {arr_to_share.nbytes} bytes')
            self.metadata = SharedNPMeta(shmem_name=self.shmem.name, 
                                         shape=arr_to_share.shape, 
                                         dtype=arr_to_share.dtype)
            self.array = np.ndarray(arr_to_share.shape, arr_to_share.dtype, buffer=self.shmem.buf)
            self.array[:] = arr_to_share#[:]
            # logging.warning(f'just shared {self.array}')
            logging.warning(f'time to share {nbytes} {time.time() - start_time}')
        else:
            # Makes local array with given shared memory
            assert metadata is not None, "Please provide either an array_to_share or metadata"
            start_time = time.time()
            self.metadata = metadata
            # print(f'recv side shmem name {metadata.shmem_name}')
            self.shmem = shared_memory.SharedMemory(name=metadata.shmem_name)
            # logging.warning(f'getting {self.shmem.name}')
            self.array = np.ndarray(metadata.shape, dtype=metadata.dtype, buffer=self.shmem.buf)
            logging.warning(f'time to recieve {time.time() - start_time}')
        self.name = name

    def __repr__(self):
        return f'SharedNPArray: name:{self.name}, meta{self.metadata}, closed{self.closed}'

    def localize(self, close_shared=True, unlink_shared=False):
        # dump contents into local (unshared memory)
        #logging.warning(f'self array {self.array}')
        start_time = time.time()
        new_array = np.array(self.array, copy=True)
        # logging.warning(f'localizing {self.name}')
        if close_shared:
            self.close()
        if unlink_shared:
            self.unlink()
        logging.warning(f'time to localize {time.time() - start_time}')
        return new_array

    def close(self):
        if (self.closed):
            raise ValueError(f"ERROR: Trying to close an array {self.name} {self.metadata} that has already been closed here")
        self.shmem.close()
        self.closed = True
        del self.array

    def unlink(self):
        self.shmem.unlink()
    

class SharedNPDict:
    arrays: Dict[str, SharedNPArray]

    def __init__(self, dict_to_share: Optional[Dict[str, np.ndarray]]=None, metadata: Optional[Dict[str, SharedNPMeta]]=None):
        self.arrays = {}
        if dict_to_share is not None:
            # Creates shared memory array
            assert metadata is None, "Please provide either an dict_to_share or metadata"
            assert isinstance(dict_to_share, dict), f"Dict to share must be a dictionary. got {type(dict_to_share)}"

            for k, v in dict_to_share.items():
                self.arrays[k] = SharedNPArray(arr_to_share=v, name=k)
        else:
            # Makes local array with given shared memory
            assert metadata is not None, "Please provide either an array_to_share or metadata"
            for k, v in metadata.items():
                shared_arr = SharedNPArray(metadata=v, name=k)
                self.arrays[k] = shared_arr

    def __repr__(self):
        out_dict = {}
        for k, v in self.arrays.items():
            out_dict[k] = str(v)
        return str(out_dict)

    def localize(self, close_shared=False, unlink_shared=False):
        # dump contents into local (unshared memory)
        # logging.warning(f'I am {self.__repr__()}')
        out_dict = {}
        for k, v in self.arrays.items():
            local_arr = v.localize(close_shared=close_shared, unlink_shared=unlink_shared)
            out_dict[k] = local_arr
        return out_dict

    def close(self):
        for _, v in self.arrays.items():
            v.close()

    def unlink(self):
        for _, v in self.arrays.items():
            v.unlink()
    
    def close_and_unlink(self):
        for _, v in self.arrays.items():
            v.close()
            v.unlink()

    def get_metas(self):
        meta_dict = {}
        for k, v in self.arrays.items():
            meta_dict[k] = v.metadata
        return meta_dict
