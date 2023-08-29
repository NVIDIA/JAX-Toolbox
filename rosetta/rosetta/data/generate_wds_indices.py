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

import argparse
import os

import wds2idx
from absl import logging
from braceexpand import braceexpand


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    description='A script to generate the webdataset index files to be used during data loading.',
)
parser.add_argument('--archive', help='path to .tar files.')
parser.add_argument(
    '--index_dir',
    help='location to store index files',
)

args = parser.parse_args()

## make sure to generate train and eval inidces separately
os.makedirs(args.index_dir, exist_ok=True)
urls = list(braceexpand(args.archive))
for (i, url) in enumerate(urls):
    creator = wds2idx.IndexCreator(url, os.path.join(args.index_dir, f'idx_{i}.txt'))
    creator.create_index()
    creator.close()

logging.info(f'Done! Index files written to {args.index_dir}.')
