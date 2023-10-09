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

import dataclasses
import functools
import logging
import os
import pickle as pkl
import random
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union, Iterable

import jax
from jax import tree_util
import seqio
import t5.data
import tensorflow as tf
import webdataset as wds
from pytriton.client import ModelClient
import rosetta.data.multiloader as multiloader
import braceexpand

seqio_vocab = t5.data.get_default_vocabulary()
server_list = None

# -----------------------------------------------------------------------------
# Configurations
# -----------------------------------------------------------------------------
    
@dataclasses.dataclass
class ModalityConfig:
    ftype: Optional[str]
    out_type: Tuple[Any]
    shape: Tuple[int]
    process_func: Optional[Callable]
    prefilter_func: Optional[Callable]=None
    no_load: bool = False # Don't load modality from webdataset pipeline. Used for modalities that are created from others' process_funcs

@dataclasses.dataclass
class WebDatasetConfig:
  """Configuration for loading a WebDataset"""
  mixture_or_task_name: Union[str, Iterable[str]]
  batch_size: int
  shuffle: bool
  seed: Optional[int]

  # Controls total number of samples. Ignored in training
  samples: Optional[int] = None
  modalities: Optional[Mapping[str, ModalityConfig]] = None #will error if None
  batch_proc: Optional[Callable] = None
  hostnames_file: Optional[str] = None
  num_parallel_processes: int = 16
  pack=False    # Webdataset doesn't currently support text packing

# -----------------------------------------------------------------------------
# Data processing utils
# -----------------------------------------------------------------------------

# ------------------
# Image
# ------------------

def image_crop_scale(image, out_img_shape=(32, 32, 3), nhwc=True):
    """
    Resizes image to out_img_shape by first doing an aspect-preserving resize 
    to match the short side of the image to the out_img_shape, then doing a
    random crop. 

    Assumes image is ranged [0, 1]
    """
    if not nhwc:
        # if non nhwc output is desired, out_img_shape should be given in nchw format.
        # We transpose it here to make it compatible with processing and transpose later
        out_img_shape = (out_img_shape[1], out_img_shape[2], out_img_shape[0])
    curr_img_shape = image.shape
    
    # square crop randomly with min dimension. 
    min_dim = min(curr_img_shape[:2])

    left = random.randint(0, curr_img_shape[1] - min_dim)
    right = left + min_dim
    top = random.randint(0, curr_img_shape[0] - min_dim)
    bottom = top + min_dim
    image = image[top:bottom, left:right]

    #resize to final dimensions
    image = np.asarray(Image.fromarray((image * 255).astype(np.uint8)).resize(out_img_shape[:2], resample=Image.BILINEAR)) / 255.

    image = image * 2 - 1 # [-1, 1] ranging
    if not nhwc:
        image = np.transpose(image, (2, 0, 1))
    return image

def image_crop_scale_with_lowres(image, out_img_shape=(32, 32, 3), low_res_img_shape=(32, 32, 3), nhwc=True):
    """
    Resizes image to out_img_shape by first doing an aspect-preserving resize 
    to match the short side of the image to the out_img_shape, then doing a
    random crop. 
    Further returns a downsized version of this final image for superresolution training

    Assumes image is ranged [0, 1]
    """
    if not nhwc:
        # if non nhwc output is desired, out_img_shape should be given in nchw format.
        # We transpose it here to make it compatible with processing and transpose later
        out_img_shape = (out_img_shape[1], out_img_shape[2], out_img_shape[0])
        low_res_img_shape = (low_res_img_shape[1], low_res_img_shape[2], low_res_img_shape[0])
    curr_img_shape = image.shape
    
    # square crop randomly with min dimension. 
    min_dim = min(curr_img_shape[:2])

    left = random.randint(0, curr_img_shape[1] - min_dim)
    right = left + min_dim
    top = random.randint(0, curr_img_shape[0] - min_dim)
    bottom = top + min_dim
    image = image[top:bottom, left:right]

    #resize to final dimensions
    image = Image.fromarray((image * 255).astype(np.uint8)).resize(out_img_shape[:2], resample=Image.BILINEAR)
    image_large = np.asarray(image) / 255.
    image_lowres = np.asarray(image.resize(low_res_img_shape[:2], resample=Image.BILINEAR)) / 255.

    image_large = image_large * 2 - 1 # [-1, 1] ranging
    image_lowres = image_lowres * 2 - 1 # [-1, 1] ranging
    if not nhwc:
        image_large = np.transpose(image_large, (2, 0, 1))
        image_lowres = np.transpose(image_lowres, (2, 0, 1))
    return {'samples': image_large, 'low_res_images': image_lowres}

def image_subcrop_scale_with_lowres(image, init_image_shape=(1024,1024,3), crop_shape=(256,256,3), low_res_img_shape=(64,64,3), nhwc=True):
    """
    Does a random crop of an image by first resizing it to a target resolution then doing a random crop.
    Further returns a downsized version of this final image for superresolution training

    Assumes image is ranged [0, 1]
    """

    image = image_crop_scale(image, out_img_shape=init_image_shape, nhwc=nhwc)
    if not nhwc:
        # if non nhwc output is desired, out_img_shape should be given in nchw format.
        # We transpose it here to make it compatible with processing and transpose later
        out_img_shape = (out_img_shape[1], out_img_shape[2], out_img_shape[0])
        low_res_img_shape = (low_res_img_shape[1], low_res_img_shape[2], low_res_img_shape[0])
        crop_shape = (crop_shape[1], crop_shape[2], crop_shape[0])
    curr_img_shape = image.shape

    # square crop randomly
    # min_dim = min(curr_img_shape[:2])
    crop_width = crop_shape[1]
    crop_height = crop_shape[0]

    left = random.randint(0, curr_img_shape[1] - crop_width)
    right = left + crop_width
    top = random.randint(0, curr_img_shape[0] - crop_height)
    bottom = top + crop_height
    image_large = image[top:bottom, left:right]

    #resize to final dimensions
    image = Image.fromarray((image_large * 255).astype(np.uint8))#.resize(out_img_shape[:2], resample=Image.BILINEAR)
    image_lowres = np.asarray(image.resize(low_res_img_shape[:2], resample=Image.BILINEAR)) / 255.

    image_large = image_large * 2 - 1 # [-1, 1] ranging
    image_lowres = image_lowres * 2 - 1 # [-1, 1] ranging
    if not nhwc:
        image_large = np.transpose(image_large, (2, 0, 1))
        image_lowres = np.transpose(image_lowres, (2, 0, 1))
    return {'samples': image_large, 'low_res_images': image_lowres}

def blank_image(out_img_shape=(32, 32, 3), nhwc=True):
    """ Dummy image creator. Used for sampling and dummy datasets """
    img = np.zeros(out_img_shape)
    if not nhwc:
        return np.transpose(img, (2, 0, 1))
    return img

def filter_lowres(image, min_dims=(64,64,3), nhwc=True):
    # returns false for images that don't meet the minimum dimensions specified.
    # min_dims should be specified in the same format as images (NHWC or NCHW)
    shape = image.shape
    assert len(shape) == len(min_dims), f"Minimum dimension spec and image shape length need to match.\
                                         Given min_dims {min_dims} and image shape {shape}"
    if not nhwc:
        dims = [1] * len(min_dims)
        dims[0] = min_dims[1]
        dims[1] = min_dims[2]
        dims[2] = min_dims[0]
        min_dims = tuple(dims)

    for i in range(len(shape)):
        if shape[i] < min_dims[i]:
            return False

    return True

# ------------------
# Text
# ------------------

def triton_textencode(text_batch: List[str]):
    """ Encodes a list of python strings into numpy character arrays """
    enc = np.array([[np.char.encode(i, 'utf-8')] for i in text_batch])
    enc = np.reshape(enc, (enc.shape[0], 1))
    return enc

def seqio_tokenizer(shape=(128,)):
    tok_config = {'text': seqio.Feature(vocabulary=t5.data.get_default_vocabulary(), add_eos=True)}
    def tok(text):
        unpad = seqio.preprocessors.tokenize_impl({'text': text}, tok_config, copy_pretokenized=False, with_eos=True)['text']
        padded = tf.pad(unpad, [[0, shape[0] - unpad.shape[0]]])
        return padded
    return tok

def mscoco_text_process(text_in, shape, vocab=seqio_vocab):
    text = text_in['caption']

    mask = np.zeros(shape[0]) #dummy mask
    if text is None or not isinstance(text, str):
        text = ''

    return {'text': text, 'text_mask': mask}

def bare_txt_process(text, shape):
    mask = np.ones(shape[0]) #dummy mask
    if text is None or not isinstance(text, str):
        text = ''
        logging.info("WARNING: no text")
    return {'text': text, 'text_mask': mask}

def sd_clip_text_tokenize(string, tokenizer):
    tok = np.array(tokenizer(string, max_length=tokenizer.model_max_length, padding="max_length", truncation=True).input_ids)
    return {'input_ids': tok}

def cls_process(cls):
    return {'cls': cls}

# ------------------
# External inference
# ------------------

def dummy_batch_infer(batch, server_list=None, text_emb_shape:Tuple=(256, 4096), model_name:str='t5_xxl'):
    IS_BATCHED = True
    if not isinstance(batch['text'], list):
        batch_dim = 1
        IS_BATCHED = False
    else:
        batch_dim = len(batch['text'])

    # construct text masks and padding
    mask = np.zeros((batch_dim, text_emb_shape[0])).astype('int32')
    rand_idxs = np.random.randint(text_emb_shape[0], size=batch_dim)
    padded_embeds = np.random.normal(size=(batch_dim, *text_emb_shape)).astype('float32')
    for i in range(rand_idxs.shape[0]):
        mask[i, :rand_idxs[i]] = 1
        padded_embeds[i, rand_idxs[i]:] = 0

    if not IS_BATCHED:
        padded_embeds = padded_embeds[0]
        mask = mask[0] 

    batch['text'] = padded_embeds
    batch['text_mask'] = mask
    
    return batch


def batch_infer_extern(batch, server_list=None, text_emb_shape:Tuple=(128, 4096), model_name:str='t5_xxl'):
    """ 
    Calls a remote PyTriton server to get logits from a text transformer.
    This is done batched for efficiency 
    """
    if server_list is None:
        raise ValueError("server_list is required")

    text_batch = batch['text']
    IS_BATCHED=True

    full_text = text_batch
    if not isinstance(full_text, list):
        IS_BATCHED=False
        full_text = [full_text]

    # encoding text batch for triton inference
    encoded_batch = triton_textencode(full_text)

    recieved_out = False
    try_ctr = 0

    # keep trying until successful inference
    padded_embeds = None
    mask = None
    while not recieved_out or padded_embeds is None:
        rand_server = server_list[random.randint(0, len(server_list) - 1)]
        with ModelClient(rand_server, model_name=model_name) as client:
            try:
                text_emb_dict = client.infer_batch(encoded_batch)
            except:
                text_emb_dict = None

        if text_emb_dict:
            #embeds = [pkl.loads(inst) for inst in text_emb_dict['encodings']]
            seqlens = text_emb_dict['encodings_seqlens'] 
            batch_dim = seqlens.shape[0]
            padded_embeds = text_emb_dict['encodings_padded']
            # padding embeds up to full seqlen
            if padded_embeds.shape[1] < text_emb_shape[0]:
                embeds = np.zeros((batch_dim, *text_emb_shape), dtype='float16')
                embeds[:, :padded_embeds.shape[1]] = padded_embeds
                padded_embeds = embeds
            mask = np.zeros((batch_dim, text_emb_shape[0])).astype('int32')
            for idx in range(batch_dim):
                mask[idx, :seqlens[idx]] = 1
        
        if padded_embeds is None:
            try_ctr += 1
            sam = text_batch[0]
            logging.info(f"Server returned nothing. Retrying. Attempt {try_ctr}. {sam}")
        else:
            recieved_out = True

    assert padded_embeds is not None and mask is not None
    if padded_embeds.shape[0] != len(full_text):
        logging.info("WARNING: somehow, embeds and full_text don't match!")

    # construct text masks and padding
    if not IS_BATCHED:
        padded_embeds = padded_embeds[0]
        mask = mask[0] 

    batch['text'] = padded_embeds
    batch['text_mask'] = mask
    
    return batch

# -----------------------------------------------------------------------------
# WebDataset construction and setup
# -----------------------------------------------------------------------------

def type_proc(dtype:str):
    if dtype == 'float32':
        return np.float32
    elif dtype == 'int':
        return np.int32
    elif dtype == 'float16':
        return np.float16
    elif dtype == 'bfloat16':
        return jax.numpy.bfloat16
    else:
        raise ValueError("Could not parse dtype: %s" % dtype)

def run_preproc(sample:Any, keys:List[str]=[], modalities: Mapping[str, ModalityConfig]={}):
    datapoint = {} 

    non_loaded_ctr = 0
    for i in range(len(keys)):
        k = keys[i]
        process_fn = modalities[k].process_func
        if modalities[k].no_load:
            non_loaded_ctr += 1
        elif process_fn is not None and modalities[k].ftype is None: # for generating a fixed shape dummy sample
            mod_out = process_fn()
            unpack_assign_or_assign(key=k, value=mod_out, dictionary=datapoint)
            non_loaded_ctr += 1
        else:
            mod_out = process_fn(sample[i - non_loaded_ctr]) if process_fn is not None else sample[i - non_loaded_ctr]
            unpack_assign_or_assign(key=k, value=mod_out, dictionary=datapoint)

    return datapoint

def run_prefilter(sample:Any, keys:List[str]=[], modalities: Mapping[str, ModalityConfig]={}):
    datapoint = {}

    non_loaded_ctr = 0
    for i in range(len(keys)):
        k = keys[i]
        prefilter_fn = modalities[k].prefilter_func
        if modalities[k].no_load:
            non_loaded_ctr += 1
        elif prefilter_fn is not None:
            if not prefilter_fn(sample[i - non_loaded_ctr]):
                return False
    return True

def unpack_assign_or_assign(key: Any, value: Any, dictionary: Dict, strict: bool = True):
    '''Tries to unpack value ignoring key, if value is a dictionary; or assigns value to key as fallback'''
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if strict and nested_key in dictionary:
                raise ValueError(f'strict=True, and key={nested_key} already in dictionary={dictionary}')
            dictionary[nested_key] = nested_value
    else:
        if strict and key in dictionary:
            raise ValueError(f'strict=True, and key={key} already in dictionary={dictionary}')
        dictionary[key] = value

def dict_batch(samples):
    """ batches elements of the same key """
    outer = tree_util.tree_structure([0 for _ in samples])
    inner = tree_util.tree_structure(samples[0])
    return tree_util.tree_transpose(outer, inner, samples)

def _simple_map(data, f, handler=wds.filters.reraise_exception):
    """Map samples."""
    for sample in data:
        try:
            result = f(sample)
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break
        if result is None:
            continue
        yield result

simple_map = wds.filters.pipelinefilter(_simple_map)

def url_process(url_str:Union[str, Iterable[str]]) -> Union[List[str], str]:
    """
    If url_str is a directory, this function will return a list of 
    all .tar files found recursively in it.
    If url_str is an iterable, expands all urls contained as directories 
    or braceexpands and concatenates them together
    """
    logging.info(f'Processing URLS for {url_str}')
    if isinstance(url_str, str):
        url_str = [url_str]
    paths = []
    for url in url_str:
        if os.path.isdir(url):
            url_paths = [str(p) for p in Path(url).rglob('*.tar')]
            logging.info("{} tarfiles found in {}".format(len(url_paths), url))
            paths += url_paths
        else:
            logging.info(f'{url} doesn\'t seem to be a directory. Treating it as a path with braceexpand')
            paths += list(braceexpand.braceexpand(url))
    return paths

def get_mm_wds_from_urls(cfg: WebDatasetConfig, batch_size:int =-1) -> Tuple[Any, Mapping[str, Tuple[int]], Mapping[str, Any]]:
    global server_list 

    # Getting all urls
    urls = url_process(cfg.mixture_or_task_name)

    # Setting up modalities (shapes and types)
    modalities = cfg.modalities
    assert modalities is not None, "Modalities cannot be None. Don't know how to process data!"
    keys = list(modalities.keys())
    in_ftypes = []
    out_shapes = {}
    out_types = {}
    for k in keys:
        m = modalities[k]
        if m.ftype is not None:
            in_ftypes.append(m.ftype)
        unpack_assign_or_assign(key=k, value=m.shape, dictionary=out_shapes)
        unpack_assign_or_assign(key=k, value=jax.tree_map(type_proc, m.out_type), dictionary=out_types)

    # Inference Server determination
    if server_list is None:
        if cfg.hostnames_file not in (None, "", "None"):
            with open(cfg.hostnames_file, 'r') as f:
                server_list = f.readlines()
            for i in range(len(server_list)):
                server_list[i] = server_list[i].strip()
        else:
            logging.info("No hostnames file. Will not initialize remote inferencing")
            server_list = None
    else:
        logging.info('SERVER LIST GIVEN. Not reading from cfg.hostnames_file')
        logging.info(server_list)

    preprocessor = functools.partial(run_preproc, keys=keys, modalities=modalities)
    pre_filter = functools.partial(run_prefilter, keys=keys, modalities=modalities)
    dataset = wds.WebDataset(urls, resampled=True).shuffle(0).decode("rgb").to_tuple(*in_ftypes).select(pre_filter).map(preprocessor)
    if cfg.samples:
        dataset = dataset.with_length(cfg.samples)
    if batch_size > 0:
        dataset = dataset.batched(batch_size, collation_fn=dict_batch)
    if cfg.batch_proc is not None:
        bp = functools.partial(cfg.batch_proc, server_list=server_list)
        dataset = dataset.compose(simple_map(bp))
    
    load = dataset
    if cfg.num_parallel_processes > 1:
        load = multiloader.MultiLoader(dataset, workers=cfg.num_parallel_processes)
    return load, out_shapes, out_types

def get_random_wds(cfg: WebDatasetConfig) -> Tuple[Any, Mapping[str, Tuple[int]], Mapping[str, Any]]:
    '''
    same as get_mm_wds_from_urls, except random dataset for SOL test or if you don't have a dataset yet
    '''
    # Setting up modalities (shapes and types)
    modalities = cfg.modalities
    assert modalities is not None, "Modalities cannot be None. Don't know how to process data!"
    keys = list(modalities.keys())
    in_ftypes = []
    out_shapes = {}
    out_types = {}
    for k in keys:
        m = modalities[k]
        if m.ftype is not None:
            in_ftypes.append(m.ftype)
        unpack_assign_or_assign(key=k, value=m.shape, dictionary=out_shapes)
        unpack_assign_or_assign(key=k, value=jax.tree_map(type_proc, m.out_type), dictionary=out_types)

    preprocessor = functools.partial(run_preproc, keys=keys, modalities=modalities)
    def random_generator(wds_config: WebDatasetConfig, num_elements: int = 100):
        for _ in range(num_elements):
            datum = {}
            for _, modality_config in wds_config.modalities.items():
                if isinstance(modality_config.shape, (tuple, list)):
                    datum[modality_config.ftype] = np.random.randint(size=modality_config.shape, low=0, high=2).astype(modality_config.out_type)
                else:
                    datum[modality_config.ftype] = jax.tree_map(
                        lambda shape, dtype: np.random.randint(size=shape, low=0, high=2).astype(dtype),
                        modality_config.shape,
                        modality_config.out_type,
                        is_leaf=lambda shape: isinstance(shape, (list, tuple)),
                    )
            yield datum

    dataset = wds.DataPipeline(
        lambda: random_generator(cfg),
        wds.to_tuple(*in_ftypes),
        wds.map(preprocessor),
    )
    print(len(list(dataset)), 'a')
    if cfg.batch_size > 0:
        dataset.pipeline.append(wds.batched(cfg.batch_size, collation_fn=dict_batch))
    if cfg.batch_proc is not None:
        bp = functools.partial(cfg.batch_proc, server_list=server_list)
        dataset = dataset.compose(simple_map(bp))
    print(len(list(dataset)), 'b')

    if cfg.num_parallel_processes > 1:
        raise NotImplementedError(f'No suport for parallel processes for random data generation')
    return dataset, out_shapes, out_types