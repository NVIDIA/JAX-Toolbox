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

from utils import ConvertHelper

CATAGORIES = ['target', "state.param_states.1.0.mu", "state.param_states.1.0.nu"]


class T5XConvertHelperBase(ConvertHelper):

    @property
    def catagories(self):
        if self.weight_only:
            return CATAGORIES[:1]
        return CATAGORIES


class T5X2TENotFuseQKVConvertHelper(T5XConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        embed_dim = self.model_config.embed_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"encoder.layers_{i}.pre_attention_layer_norm.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.query.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.query.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.query.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.key.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.key.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.value.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.value.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.pre_mlp_layer_norm.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.mlp.wi_0.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_kernel",
                                          (embed_dim, mlp_intermediate_dim),
                                          None,
                                          extra_src_paths=[f"encoder.layers_{i}.mlp.wi_1.kernel"],
                                          stack_dim=1),
                f"encoder.layers_{i}.mlp.wo.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wo_kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.pre_self_attention_layer_norm.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.query.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.query.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.query.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.key.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.key.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.value.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.value.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.pre_cross_attention_layer_norm.scale":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.scale",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.query.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.key.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.key.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.value.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.value.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.pre_mlp_layer_norm.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.mlp.wi_0.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_kernel",
                                          (embed_dim, mlp_intermediate_dim),
                                          None,
                                          extra_src_paths=[f"decoder.layers_{i}.mlp.wi_1.kernel"],
                                          stack_dim=1),
                f"decoder.layers_{i}.mlp.wo.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wo_kernel",
                                          None,
                                          None,
                                          just_copy=True),
            })
        return ckpt_map


class TENotFuseQKV2T5XConvertHelper(T5XConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        embed_dim = self.model_config.embed_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"encoder.layers_{i}.attention.query.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.pre_attention_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.query.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.query.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.key.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.key.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.value.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.value.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.mlp.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.pre_mlp_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.mlp.wi_kernel": [
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_0.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_1.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 1, :])
                ],
                f"encoder.layers_{i}.mlp.wo_kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wo.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.query.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.pre_self_attention_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.query.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.query.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.key.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.key.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.value.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.value.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.query.scale":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.pre_cross_attention_layer_norm.scale",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.query.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.key.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.key.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.value.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.value.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.mlp.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.pre_mlp_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.mlp.wi_kernel": [
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_0.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_1.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 1, :]),
                ],
                f"decoder.layers_{i}.mlp.wo_kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wo.kernel",
                                          None,
                                          None,
                                          just_copy=True),
            })
        return ckpt_map


class T5X2TEFuseQKVConvertHelper(T5XConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        embed_dim = self.model_config.embed_dim
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"encoder.layers_{i}.pre_attention_layer_norm.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.qkv.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.query.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.qkv.kernel",
                                          (embed_dim, hidden_dim),
                                          None,
                                          extra_src_paths=[
                                              f"encoder.layers_{i}.attention.key.kernel",
                                              f"encoder.layers_{i}.attention.value.kernel"
                                          ],
                                          stack_dim=1),
                f"encoder.layers_{i}.pre_mlp_layer_norm.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.mlp.wi_0.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_kernel",
                                          (embed_dim, mlp_intermediate_dim),
                                          None,
                                          extra_src_paths=[f"encoder.layers_{i}.mlp.wi_1.kernel"],
                                          stack_dim=1),
                f"encoder.layers_{i}.mlp.wo.kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wo_kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.pre_self_attention_layer_norm.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.qkv.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.query.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.qkv.kernel",
                                          (embed_dim, hidden_dim),
                                          None,
                                          extra_src_paths=[
                                              f"decoder.layers_{i}.self_attention.key.kernel",
                                              f"decoder.layers_{i}.self_attention.value.kernel"
                                          ],
                                          stack_dim=1),
                f"decoder.layers_{i}.pre_cross_attention_layer_norm.scale":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.scale",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.query.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.key.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.kv.kernel",
                        (embed_dim, hidden_dim),
                        None,
                        extra_src_paths=[
                            f"decoder.layers_{i}.encoder_decoder_attention.value.kernel"
                        ],
                        stack_dim=1),
                f"decoder.layers_{i}.pre_mlp_layer_norm.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.mlp.wi_0.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_kernel",
                                          (embed_dim, mlp_intermediate_dim),
                                          None,
                                          extra_src_paths=[f"decoder.layers_{i}.mlp.wi_1.kernel"],
                                          stack_dim=1),
                f"decoder.layers_{i}.mlp.wo.kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wo_kernel",
                                          None,
                                          None,
                                          just_copy=True),
            })
        return ckpt_map


class TEFuseQKV2T5XConvertHelper(T5XConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        embed_dim = self.model_config.embed_dim
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"encoder.layers_{i}.attention.qkv.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.pre_attention_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.attention.qkv.kernel": [
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.query.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.key.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 1, :]),
                    self._get_convert_pkg(f"encoder.layers_{i}.attention.value.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 2, :])
                ],
                f"encoder.layers_{i}.mlp.scale":
                    self._get_convert_pkg(f"encoder.layers_{i}.pre_mlp_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"encoder.layers_{i}.mlp.wi_kernel": [
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_0.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wi_1.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 1, :])
                ],
                f"encoder.layers_{i}.mlp.wo_kernel":
                    self._get_convert_pkg(f"encoder.layers_{i}.mlp.wo.kernel",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.qkv.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.pre_self_attention_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.self_attention.qkv.kernel": [
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.query.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.key.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 1, :]),
                    self._get_convert_pkg(f"decoder.layers_{i}.self_attention.value.kernel",
                                          (embed_dim, 3, hidden_dim), None, lambda x: x[:, 2, :])
                ],
                f"decoder.layers_{i}.encoder_decoder_attention.query.scale":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.pre_cross_attention_layer_norm.scale",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.query.kernel":
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.query.kernel",
                        None,
                        None,
                        just_copy=True),
                f"decoder.layers_{i}.encoder_decoder_attention.kv.kernel": [
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.key.kernel",
                        (embed_dim, 2, hidden_dim), None, lambda x: x[:, 0, :]),
                    self._get_convert_pkg(
                        f"decoder.layers_{i}.encoder_decoder_attention.value.kernel",
                        (embed_dim, 2, hidden_dim), None, lambda x: x[:, 1, :])
                ],
                f"decoder.layers_{i}.mlp.scale":
                    self._get_convert_pkg(f"decoder.layers_{i}.pre_mlp_layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"decoder.layers_{i}.mlp.wi_kernel": [
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_0.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 0, :]),
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wi_1.kernel",
                                          (embed_dim, 2, mlp_intermediate_dim), None,
                                          lambda x: x[:, 1, :]),
                ],
                f"decoder.layers_{i}.mlp.wo_kernel":
                    self._get_convert_pkg(f"decoder.layers_{i}.mlp.wo.kernel",
                                          None,
                                          None,
                                          just_copy=True),
            })
        return ckpt_map
