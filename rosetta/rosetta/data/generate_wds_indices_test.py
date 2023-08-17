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

import os
import sys


def test_generated_indices(
    dummy_wds_metadata,
    run_subprocess_blocking,
    tmp_path,
):
    index_dir = f'{tmp_path}/indices'

    stdout, stderr, returncode = run_subprocess_blocking(
        sys.executable, '-m',
        'rosetta.data.generate_wds_indices',
        f'--archive={dummy_wds_metadata.path}',
        f'--index_dir={index_dir}')

    files = os.listdir(index_dir)
    print(index_dir)

    ## one index file per wds tar file
    assert len(files)==1

    with open(os.path.join(index_dir, files[0])) as test_file:
      lines = test_file.readlines()

      assert len(lines) == dummy_wds_metadata.num_examples+1

      first_example = lines[1].split()
      ### 4 entries per modality, 3 modalities (cls, jpg, txt)
      assert len(first_example) == 12

      final_example = lines[-1].split()
      assert final_example[0] == 'cls'
      assert final_example[3] == 'sample000079.cls'
