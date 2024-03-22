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

## TODO: support GQA!

import jax.numpy as jnp

from utils import ConvertHelper

class PaxConvertHelperBase(ConvertHelper):

    @property
    def catagories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]


class NonRepeat2RepeatConvertHelper(PaxConvertHelperBase):
    
    @property
    def catagories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]

    def _generate_ckpt_map(self):
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim
        
        ckpt_map = {f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.ffn_layer1.linear.w",
                            (hidden_dim, mlp_intermediate_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.linear.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.ffn_layer2.linear.w",
                            (mlp_intermediate_dim, hidden_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.linear.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.layer_norm.scale",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.scale" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.layer_norm.scale":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.layer_norm.scale",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.layer_norm.scale" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.self_attention.combined_qkv.w",
                            (3, hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.self_attention.post.w":
                      self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.self_attention.post.w",
                            (hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.post.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                   }
        
        if not self.skip_bias:
                ckpt_map.update({
                    f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.ffn_layer1.bias.b",
                            (mlp_intermediate_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.bias.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.ffn_layer2.bias.b",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.bias.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.ff_layer.layer_norm.bias",
                            (mlp_intermediate_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.bias" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.layer_norm.bias":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.layer_norm.bias",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.layer_norm.bias" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.self_attention.combined_qkv.b",
                            (3, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.repeat.sub.x_layers_0.self_attention.post.b":
                        self._get_convert_pkg(
                            f"lm.transformer.x_layers_0.self_attention.post.b",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.post.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                })
                
        return ckpt_map