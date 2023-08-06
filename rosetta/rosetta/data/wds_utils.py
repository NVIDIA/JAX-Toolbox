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

import dataclasses
from typing import Any


# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------

@dataclasses.dataclass
class ModalityConfig:
    """Information about a particular modality present in the dataset.
       Note that a 1-1 mapping is expected between modalities and features,
       such that a single modality maps to a single feature in the dataset.

       Attributes:
         name: Desired name of the feature to be created using this modality.
         ftype: Extension of the files names corresponding to this modality .
           (e.g. 'jpg', 'png', 'cls', 'txt')
         out_type: Expected return type for this feature.
         shape: Expected output shape for this feature.
    """

    name: str | None
    ftype: str | None
    out_type: tuple[Any]
    shape: tuple[int]

@dataclasses.dataclass
class WebDatasetConfig:
  """Configuration for loading a WebDataset

     Attributes:
       urls: String with the path to the webdataset tar files. A sequence of tar files
         can be specified using braceexpand notation.
       batch_size: Global batch size.
       seed: Dataloading seed.
       index_dir: Path to the index files corresponding to the webdataset. Index files can be
         generated using `generate_wds_indices.py`.
       prefetch: Prefetch depth of the dataloading pipeline
       shuffle: Whether to shuffle the data
       num_parallel_processes: Number of CPU threads used for the dataloading pipeline.
  """
  urls: str
  batch_size: int
  seed: int | None

  index_dir: str | None = None
  prefetch: int = 2  # prefetch buffer size
  shuffle: bool = True
  num_parallel_processes: int = 16

