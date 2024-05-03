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

from paxml_converters import (
    NonRepeat2RepeatConvertHelper,
    Repeat2NonRepeatConvertHelper
)
from ..common.utils import ModelConfig

PAX_CONVERT_HELPER_DICT = {
    "to_repeat": NonRepeat2RepeatConvertHelper,
    "from_repeat": Repeat2NonRepeatConvertHelper,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Pax Non-repeat to repeat CKPT Converter.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input-path',
                        type=str,
                        required=True,
                        help="the path to load a source checkponint for this conversion.")
    parser.add_argument('--output-path',
                        type=str,
                        required=True,
                        help="the path to store the converted checkponint.")
    parser.add_argument('--direction',
                        type=str,
                        choices=("to_repeat", "from_repeat"),
                        required=True,
                        help="the framework that stored the given source checkpoint.")
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
    parser.add_argument('--weight-only',
                        action="store_true",
                        default=False,
                        help="indicate if the source checkpoint only includes weights.")
    parser.add_argument('--skip-bias',
                        action="store_true",
                        default=False,
                        help="indicate whether the source checkpoint has biases.")
    parser.add_argument(
        '--use-gated-activations',
        action="store_true",
        default=False,
        help="indicate if the model uses a gated activation function.")
    parser.add_argument(
        '--split-qkv',
        action="store_true",
        default=False,
        help="indicate if the source Pax checkpoint has split QKV parameters.")
    parser.add_argument(
        '--skip-position-emb',
        action="store_true",
        default=False,
        help="indicate if the model does NOT have a trainable position embedding.")

    args = parser.parse_args()

    return args


def get_convert_helper(args):

    model_config = ModelConfig(args.num_of_layer,
                               None, # embed_dim unused
                               args.num_of_head,
                               args.head_dim,
                               args.mlp_intermediate_dim)

    return PAX_CONVERT_HELPER_DICT[args.direction](
        input_path=args.input_path,
        output_path=args.output_path,
        model_config=model_config,
        weight_only=args.weight_only,
        skip_bias=args.skip_bias,
        use_gated_activations=args.use_gated_activations,
        split_qkv=args.split_qkv,
        skip_position_emb=args.skip_position_emb
    )


if __name__ == "__main__":
    args = parse_args()
    convert_helper = get_convert_helper(args)
    convert_helper.convert()
