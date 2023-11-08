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

import sys


def print_output(stdout, stderr):
    def box_print(text):
        n_len = len(text)
        print('='*(n_len + 4))
        print(f'= {text} =')
        print('='*(n_len + 4))
    box_print('stdout')
    print(stdout.decode())
    box_print('stderr')
    print(stderr.decode())


def test_small_vit_train_on_dummy_data(dummy_wds_metadata, run_subprocess_blocking, package_root_dir, tmp_path):
    tmp_model_dir = str(tmp_path)
    stdout, stderr, returncode = run_subprocess_blocking(
        sys.executable, '-m',
        't5x.train',
        '--gin_file=rosetta/projects/vit/configs/tests/small_pretrain_dummy.gin',
        '--gin.TRAIN_STEPS=100',
        f'--gin.MIXTURE_OR_TASK_NAME="{dummy_wds_metadata.path}"',
        f'--gin.MODEL_DIR="{tmp_model_dir}"',
        '--gin.DTYPE="bfloat16"',
        '--gin.BATCH_SIZE=4',
        '--gin.train.stats_period=100',
        '--gin.trainer.Trainer.num_microbatches=0',
        '--gin_search_paths=/opt/rosetta',
        env={'CUDA_VISIBLE_DEVICES': '0'},
    )
    print_output(stdout, stderr)
    assert returncode == 0
