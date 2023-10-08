'''Debugging tool: prints out the first element in a Webdataset, print cardinality, and create small subset for testing'''
import argparse
import json
import time

import jax
import numpy as np
import tqdm
import webdataset as wds
from jax import tree_util


def list_of_dict_to_dict_of_list(samples):
    outer = tree_util.tree_structure([0 for _ in samples])
    inner = tree_util.tree_structure(samples[0])
    return tree_util.tree_transpose(outer, inner, samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('urls', type=str, help='Urls for webdataset. Supports braceexpand syntax')
    parser.add_argument('--len', default=False, action='store_true')
    parser.add_argument('--batch_size', default=1, type=int, help='If provided will call batched(N) on the Webdataset')
    parser.add_argument('--first100', default=None, type=str, help='If provided, will make a sample of 100 and write it to the file named here')

    args = parser.parse_args()

    dataset = wds.WebDataset(args.urls).decode('rgb')
    if args.first100:
        if not args.first100.endswith('.tar'):
            raise ValueError(f'--first100={args.first100} should end with .tar')
        writer = wds.TarWriter(args.first100)
        for i, x in enumerate(tqdm.tqdm(dataset, desc=f'Writing to samples to {args.first100}')):
            if i == 100:
                break
            writer.write(x)
        writer.close()
    single_elem = next(iter(dataset))
    keys = list(single_elem.keys())

    if args.batch_size > 1:
        dataset = dataset.to_tuple(*keys).batched(args.batch_size, collation_fn=list_of_dict_to_dict_of_list)

    def printer(obj):
        if isinstance(obj, (str, bytes, int, float)):
            return obj
        elif isinstance(obj, np.ndarray):
            return f'np.ndarray(shape={obj.shape}, elem[:3]={np.ravel(obj)[:3]})'
        else:
            raise ValueError(f'Not sure how to print type {type(obj)}: {obj}')
    print('== SINGLE EXAMPLE ==')
    print(json.dumps(jax.tree_map(printer, single_elem), indent=2))
    # if args.batch_size > 1:
    #     print(f'== BATCH [N={args.batch_size}] EXAMPLE ==')
    #     print(json.dumps(jax.tree_map(printer, next(iter(dataset))), indent=2))
    if args.len:
        start = time.time()
        for i, x in enumerate(tqdm.tqdm(dataset, desc='iterating thru dataset for len')):
            pass
        elapsed = time.time() - start
        print(f'Dataset length: {i+1}')
        print(f'example/sec: {args.batch_size*(i+1)/elapsed:.3f}')
        print(f'batch/sec: {(i+1)/elapsed:.3f}')
