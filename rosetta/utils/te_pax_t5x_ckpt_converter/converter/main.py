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

from paxml_converters import Pax2TEConvertHelper, Pax2TERepeatConvertHelper
from paxml_converters import TE2PaxConvertHelper, TE2PaxRepeatConvertHelper
from t5x_converters import T5X2TEFuseQKVConvertHelper, T5X2TENotFuseQKVConvertHelper
from t5x_converters import TEFuseQKV2T5XConvertHelper, TENotFuseQKV2T5XConvertHelper
from utils import ModelConfig

PAX = 'pax'
T5X = 't5x'

FW2TE = 'fw2te'
TE2FW = 'te2fw'

# Key = (Direction, isRepeat)
PAX_CONVERT_HELPER_DICT = {
    (FW2TE, False): Pax2TEConvertHelper,
    (FW2TE, True): Pax2TERepeatConvertHelper,
    (TE2FW, False): TE2PaxConvertHelper,
    (TE2FW, True): TE2PaxRepeatConvertHelper,
}

# Key = (Direction, isFusedQKV)
T5X_CONVERT_HELPER_DICT = {
    (FW2TE, False): T5X2TENotFuseQKVConvertHelper,
    (FW2TE, True): T5X2TEFuseQKVConvertHelper,
    (TE2FW, False): TENotFuseQKV2T5XConvertHelper,
    (TE2FW, True): TEFuseQKV2T5XConvertHelper,
}


def parse_args():
    parser = argparse.ArgumentParser(description="TE <-> Paxml/T5X CKPT Converter.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-path',
                        type=str,
                        required=True,
                        help="the path to load a source checkponint for this conversion.")
    parser.add_argument('--output-path',
                        type=str,
                        required=True,
                        help="the path to store the converted checkponint.")
    parser.add_argument('--fw',
                        type=str,
                        choices=(PAX, T5X),
                        required=True,
                        help="the framework that stored the given source checkpoint.")
    parser.add_argument('--direction',
                        type=str,
                        choices=(FW2TE, TE2FW),
                        required=True,
                        help="the direction of this conversion.")

    parser.add_argument('--num-of-layer',
                        type=int,
                        required=True,
                        help="the number of Transformer layer of the given source checkpoint.")
    parser.add_argument(
        '--num-of-head',
        type=int,
        required=True,
        help="the number of head of multi-head attention of the given source checkpoint.")
    parser.add_argument(
        '--head-dim',
        type=int,
        required=True,
        help="the head dimension of multi-head attention of the given source checkpoint.")
    parser.add_argument(
        '--mlp-intermediate-dim',
        type=int,
        required=True,
        help="the intermediate dimension of MLP block (FFN) of the given source checkpoint.")
    parser.add_argument(
        '--embed-dim',
        type=int,
        default=None,
        help="the embeded dimension of the given source checkpoint, must give if --fw=t5x."
        " (default is None)")
    parser.add_argument(
        '--kernel-chunk-size',
        type=int,
        default=None,
        help="the size to chucnk kernel (weighs) then store, only support with --fw=pax."
        " Setting None means no chunking.")

    parser.add_argument('--weight-only',
                        action="store_true",
                        default=False,
                        help="indicate if the source checkpoint only includes weights.")

    parser.add_argument('--skip-ln',
                        action="store_true",
                        default=False,
                        help="indicate if skip the conversion for LayerNorm.")

    parser.add_argument('--pax-repeat',
                        action="store_true",
                        default=False,
                        help="indicate if the source Pax checkpoint enables Repeat.")
    parser.add_argument(
        '--t5x-fuse-qkv',
        action="store_true",
        default=False,
        help="indicate if the source T5X checkpoint enables fused_qkv_params of TE.")

    args = parser.parse_args()

    if args.fw == T5X:
        assert args.embed_dim is not None
    return args


def get_convert_helper(args):

    model_config = ModelConfig(args.num_of_layer, args.embed_dim, args.num_of_head, args.head_dim,
                               args.mlp_intermediate_dim, args.kernel_chunk_size)

    convert_helper_cls = None

    if args.fw == PAX:
        convert_helper_cls = PAX_CONVERT_HELPER_DICT[(args.direction, args.pax_repeat)]

    if args.fw == T5X:
        convert_helper_cls = T5X_CONVERT_HELPER_DICT[(args.direction, args.t5x_fuse_qkv)]

    assert convert_helper_cls is not None, "Not Supported."
    return convert_helper_cls(args.input_path, args.output_path, model_config,
                              args.weight_only, args.skip_ln)


if __name__ == "__main__":
    args = parse_args()
    convert_helper = get_convert_helper(args)
    convert_helper.convert()
