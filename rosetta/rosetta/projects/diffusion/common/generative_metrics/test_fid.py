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

import jax
import jax.experimental.maps
import jax.numpy as jnp
import numpy as np

import torch.utils.data
from torchvision import datasets, transforms

from torch.utils.data import Dataset

from jax_multimodal.common.generative_metrics.fid_metric import fid
from jax_multimodal.common.generative_metrics.inception_v3 import load_pretrained_inception_v3
import sys
import os
import glob
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, root):
        self.image_paths = self.get_image_paths(root)
        self.transform = transforms.ToTensor()#lambda x: np.array(x)

    def get_image_paths(self,directory):
        image_paths = []
        
        # Recursive search for all image files
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_paths.append(os.path.join(root, file))
        
        return image_paths
        
    def __getitem__(self, index):
        image_path = self.image_paths[index]
        x = Image.open(image_path)
        if self.transform is not None:
            x = self.transform(x)
        if x.shape != (3, 256, 256):
            if x.shape[0] == 1:
                x = torch.cat([x,x,x], dim=0)
            if x.shape[0] == 4:
                x = x[:3]
        return x
    
    def __len__(self):
        return len(self.image_paths)

def collate_fn(batch):
  return torch.stack(batch)

TRAIN_ROOT = sys.argv[1]
TEST_ROOT = sys.argv[2]
NUM_SAMPLES = int(sys.argv[3])

def load_cifar10(batch_size=128):
    """Load CIFAR10 dataset."""

    train_dataset = MyDataset(TRAIN_ROOT)
    test_dataset = MyDataset(TEST_ROOT)
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(NUM_SAMPLES))
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(NUM_SAMPLES))
    print(len(train_dataset))
    print(len(test_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn,
    )
    return train_loader, test_loader


train, test = load_cifar10(1024)

fid(train, test, 'inception_weights', 1)
