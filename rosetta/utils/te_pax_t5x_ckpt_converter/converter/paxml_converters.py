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


import jax.numpy as jnp

from utils import ConvertHelper


class PaxConvertHelperBase(ConvertHelper):

    @property
    def catagories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]


class Pax2TEConvertHelper(PaxConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.bias.b":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wi_bias",
                        (mlp_intermediate_dim,), None, lambda x: jnp.reshape(x, (1, *x.shape))),
                f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.linear.w":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wi_kernel",
                        (hidden_dim, mlp_intermediate_dim), 0,
                        lambda x: jnp.reshape(x, (*x.shape[:-1], 1, x.shape[-1]))),
                f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.bias.b":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wo_bias",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.linear.w":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wo_kernel",
                        (mlp_intermediate_dim, hidden_dim), 1),
                f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.bias":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.ln_bias",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.scale":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.scale",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.layer_norm.bias":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.ln_bias",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.layer_norm.scale":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.scale",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.b":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.bias",
                        (3, num_of_head, head_dim), None,
                        lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1]))),
                f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.w":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.kernel",
                        (3, hidden_dim, num_of_head, head_dim), 0,
                        lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1])),
                        lambda x: jnp.transpose(x, (1, 0, 2))),
                f"lm.transformer.x_layers_{i}.self_attention.post.b":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.out.bias",
                        None,
                        None,
                        just_copy=True),
                f"lm.transformer.x_layers_{i}.self_attention.post.w":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.out.kernel",
                        (hidden_dim, num_of_head, head_dim), 1,
                        lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1])),
                        lambda x: jnp.transpose(x, (1, 0)))
            })

        return ckpt_map


class TE2PaxConvertHelper(PaxConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        for i in range(self.model_config.num_of_layer):
            ckpt_map.update({
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wi_bias":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.bias.b",
                                          (1, mlp_intermediate_dim), None,
                                          lambda x: jnp.reshape(x, (x.shape[-1],))),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wi_kernel":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.linear.w",
                        (hidden_dim, 1, mlp_intermediate_dim), 0,
                        lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-1]))),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wo_bias":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.bias.b",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.wo_kernel":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.linear.w",
                        (mlp_intermediate_dim, hidden_dim), 1),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.ln_bias":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.bias",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.mlp.scale":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.ln_bias":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.layer_norm.bias",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.scale":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.layer_norm.scale",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.bias":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.b",
                        (3, hidden_dim), None,
                        lambda x: jnp.reshape(x, (*x.shape[:-1], num_of_head, head_dim))),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.qkv.kernel":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.w",
                        (hidden_dim, 3, hidden_dim), 1,
                        lambda x: jnp.reshape(x, (*x.shape[:-1], num_of_head, head_dim)),
                        lambda x: jnp.transpose(x, (1, 0, 2, 3))),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.out.bias":
                    self._get_convert_pkg(f"lm.transformer.x_layers_{i}.self_attention.post.b",
                                          None,
                                          None,
                                          just_copy=True),
                f"lm.transformer.x_layers_{i}.transformerlayer.cld.attention.out.kernel":
                    self._get_convert_pkg(
                        f"lm.transformer.x_layers_{i}.self_attention.post.w",
                        (hidden_dim, hidden_dim), 0,
                        lambda x: jnp.reshape(x, (num_of_head, head_dim, hidden_dim)),
                        lambda x: jnp.transpose(x, (2, 0, 1))),
            })
        return ckpt_map


class PaxRepeatConvertHelperBase(ConvertHelper):

    @property
    def catagories(self):
        if self.weight_only:
            return ['mdl_vars.params']

        num_of_layer = self.model_config.num_of_layer
        return [
            'mdl_vars.params', f"opt_states_0.p#{num_of_layer}#i-1_2.m.params",
            f"opt_states_0.p#{num_of_layer}#i-1_2.v.params"
        ]


class Pax2TERepeatConvertHelper(PaxRepeatConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        num_of_layer = self.model_config.num_of_layer
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        ckpt_map.update({
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wi_bias',
                    (num_of_layer, mlp_intermediate_dim), None,
                    lambda x: jnp.reshape(x, (*x.shape[:-1], 1, x.shape[-1]))),
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wi_kernel',
                    (num_of_layer, hidden_dim, mlp_intermediate_dim), 1,
                    lambda x: jnp.reshape(x, (*x.shape[:-1], 1, x.shape[-1]))),
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wo_bias',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wo_kernel',
                    (num_of_layer, mlp_intermediate_dim, hidden_dim), 2),
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.ln_bias',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.scale',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.layer_norm.bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.ln_bias',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.layer_norm.scale':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.scale',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.bias',
                    (num_of_layer, 3, num_of_head, head_dim), None,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1]))),
            'lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.kernel',
                    (num_of_layer, 3, hidden_dim, num_of_head, head_dim), 1,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1])),
                    lambda x: jnp.transpose(x, (0, 2, 1, 3))),
            'lm.transformer.repeat.sub.x_layers_0.self_attention.post.b':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.out.bias',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.self_attention.post.w':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.out.kernel',
                    (num_of_layer, hidden_dim, num_of_head, head_dim), 2,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-2] * x.shape[-1])),
                    lambda x: jnp.transpose(x, (0, 2, 1)))
        })

        return ckpt_map


class TE2PaxRepeatConvertHelper(PaxRepeatConvertHelperBase):

    def _generate_ckpt_map(self):
        ckpt_map = {}

        num_of_layer = self.model_config.num_of_layer
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim

        ckpt_map.update({
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wi_bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b',
                    (num_of_layer, 1, mlp_intermediate_dim), None,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-1]))),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wi_kernel':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w',
                    (num_of_layer, hidden_dim, 1, mlp_intermediate_dim), 1,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], x.shape[-1]))),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wo_bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.wo_kernel':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w',
                    (num_of_layer, mlp_intermediate_dim, hidden_dim), 2),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.ln_bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.mlp.scale':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale',
                    None,
                    None,
                    just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.ln_bias':
                self._get_convert_pkg('lm.transformer.repeat.sub.x_layers_0.layer_norm.bias',
                                      None,
                                      None,
                                      just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.scale':
                self._get_convert_pkg('lm.transformer.repeat.sub.x_layers_0.layer_norm.scale',
                                      None,
                                      None,
                                      just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.bias':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b',
                    (num_of_layer, 3, hidden_dim), None,
                    lambda x: jnp.reshape(x, (*x.shape[:-1], num_of_head, head_dim))),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.qkv.kernel':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w',
                    (num_of_layer, hidden_dim, 3, hidden_dim), 2,
                    lambda x: jnp.reshape(x, (*x.shape[:-1], num_of_head, head_dim)),
                    lambda x: jnp.transpose(x, (0, 2, 1, 3, 4))),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.out.bias':
                self._get_convert_pkg('lm.transformer.repeat.sub.x_layers_0.self_attention.post.b',
                                      None,
                                      None,
                                      just_copy=True),
            'lm.transformer.repeat.sub.x_layers_0.transformerlayer.cld.attention.out.kernel':
                self._get_convert_pkg(
                    'lm.transformer.repeat.sub.x_layers_0.self_attention.post.w',
                    (num_of_layer, hidden_dim, hidden_dim), 1,
                    lambda x: jnp.reshape(x, (*x.shape[:-2], num_of_head, head_dim, hidden_dim)),
                    lambda x: jnp.transpose(x, (0, 3, 1, 2)))
        })

        return ckpt_map
