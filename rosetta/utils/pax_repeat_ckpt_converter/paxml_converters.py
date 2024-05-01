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

import jax
import jax.numpy as jnp
import functools
from utils import ConvertHelper

class PaxConvertHelperBase(ConvertHelper):

    @property
    def src_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]

    @property
    def target_categories(self):
        return self.src_categories

class NonRepeat2RepeatConvertHelper(PaxConvertHelperBase):
    
    @property
    def target_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']

        ## TODO: make num layers configurable
        return ['mdl_vars.params', f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.m.params", f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.v.params"]

    def _generate_ckpt_map(self):
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim
        num_layers = self.model_config.num_of_layer

        ckpt_map = {f"lm.transformer.x_layers_0.ff_layer.ffn_layer1.linear.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.linear.w",
                            (hidden_dim, mlp_intermediate_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.linear.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.ff_layer.ffn_layer2.linear.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.linear.w",
                            (mlp_intermediate_dim, hidden_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.linear.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.ff_layer.layer_norm.scale":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.scale",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.scale" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.layer_norm.scale":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.layer_norm.scale",
                            (hidden_dim,), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.layer_norm.scale" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.self_attention.post.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.post.w",
                            (hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.post.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                   }
        
        if not self.skip_bias:
            ckpt_map.update({
                f"lm.transformer.x_layers_0.ff_layer.ffn_layer1.bias.b":
                    self._get_convert_pkg(
                        f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1.bias.b",
                        (mlp_intermediate_dim,), 0,
                        extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1.bias.b" for i in range(1, num_layers)],
                        stack_dim = 0),
                f"lm.transformer.x_layers_0.ff_layer.ffn_layer2.bias.b":
                    self._get_convert_pkg(
                        f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer2.bias.b",
                        (hidden_dim,), 0,
                        extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer2.bias.b" for i in range(1, num_layers)],
                        stack_dim = 0),
                f"lm.transformer.x_layers_0.ff_layer.layer_norm.bias":
                    self._get_convert_pkg(
                        f"lm.transformer.repeat.sub.x_layers_0.ff_layer.layer_norm.bias",
                        (mlp_intermediate_dim,), 0,
                        extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.layer_norm.bias" for i in range(1, num_layers)],
                        stack_dim = 0),
                f"lm.transformer.x_layers_0.layer_norm.bias":
                    self._get_convert_pkg(
                        f"lm.transformer.repeat.sub.x_layers_0.layer_norm.bias",
                        (hidden_dim,), 0,
                        extra_src_paths = [f"lm.transformer.x_layers_{i}.layer_norm.bias" for i in range(1, num_layers)],
                        stack_dim = 0),
                f"lm.transformer.x_layers_0.self_attention.post.b":
                    self._get_convert_pkg(
                        f"lm.transformer.repeat.sub.x_layers_0.self_attention.post.b",
                        (hidden_dim,), 0,
                        extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.post.b" for i in range(1, num_layers)],
                        stack_dim = 0),
            })

        if self.use_gated_activations:
            ckpt_map.update({
                f"lm.transformer.x_layers_0.ff_layer.ffn_layer1_gate.linear.w":
                    self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.ff_layer.ffn_layer1_gate.linear.w",
                            (hidden_dim, mlp_intermediate_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.ff_layer.ffn_layer1_gate.linear.w" for i in range(1, num_layers)],
                            stack_dim = 0),
            })
 
        if self.split_qkv:
            ckpt_map.update({
                f"lm.transformer.x_layers_0.self_attention.query.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.query.w",
                            (hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.query.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                f"lm.transformer.x_layers_0.self_attention.key.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.key.w",
                            (hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.key.w" for i in range(1, num_layers)],
                            stack_dim = 0),
                f"lm.transformer.x_layers_0.self_attention.value.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.value.w",
                            (hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.value.w" for i in range(1, num_layers)],
                            stack_dim = 0),
            })
            if not self.skip_bias:
                ckpt_map.update({
                    f"lm.transformer.x_layers_0.self_attention.query.b":
                        self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.query.b",
                            (num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.query.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.self_attention.key.b":
                        self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.key.b",
                            (num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.key.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                    f"lm.transformer.x_layers_0.self_attention.value.b":
                        self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.value.b",
                            (num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.value.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                })
        else:
            ckpt_map.update({
                f"lm.transformer.x_layers_0.self_attention.combined_qkv.w":
                      self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.w",
                            (3, hidden_dim, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.w" for i in range(1, num_layers)],
                            stack_dim = 0),
            })
            if not self.skip_bias:
                ckpt_map.update({
                    f"lm.transformer.x_layers_0.self_attention.combined_qkv.b":
                        self._get_convert_pkg(
                            f"lm.transformer.repeat.sub.x_layers_0.self_attention.combined_qkv.b",
                            (3, num_of_head, head_dim), 0,
                            extra_src_paths = [f"lm.transformer.x_layers_{i}.self_attention.combined_qkv.b" for i in range(1, num_layers)],
                            stack_dim = 0),
                })
            
                
        return ckpt_map


    def no_prefix_conversions(self):

        num_layers = self.model_config.num_of_layer

        def repeat_count(arr, num_layers):
            return jax.numpy.repeat(arr, num_layers)

        if self.weight_only:
            return {}

        ckpt_map = {f"opt_states_0_0.count":
                        [self._get_convert_pkg(
                            f"opt_states_0.no_prefix_0.count",
                            None, None,
                            just_copy = True),
                         self._get_convert_pkg(
                            f"opt_states_0.p#{num_layers}#i-1_0.count",
                            (), 0,
                            functools.partial(repeat_count, num_layers=num_layers)),
                        ],
                    f"opt_states_0_1.count":
                        [self._get_convert_pkg(
                            f"opt_states_0.no_prefix_1.count",
                            None, None,
                            just_copy = True),
                         self._get_convert_pkg(
                            f"opt_states_0.p#{num_layers}#i-1_1.count",
                            (), 0,
                            functools.partial(repeat_count, num_layers=num_layers)),
                        ],
                    f"opt_states_0_2.count":
                        [self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.count",
                            None, None,
                            just_copy = True),
                         self._get_convert_pkg(
                            f"opt_states_0.p#{num_layers}#i-1_2.count",
                            (), 0,
                            functools.partial(repeat_count, num_layers=num_layers)),
                        ],
                    f"opt_states_0_3.count":
                        [self._get_convert_pkg(
                            f"opt_states_0.no_prefix_3.count",
                            None, None,
                            just_copy = True),
                         self._get_convert_pkg(
                            f"opt_states_0.p#{num_layers}#i-1_3.count",
                            (), 0,
                            functools.partial(repeat_count, num_layers=num_layers)),
                        ],
                    f"opt_states_0_2.m.params.lm.final_ln.scale":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.m.params.lm.final_ln.scale",
                            None, None,
                            just_copy = True),
                    f"opt_states_0_2.m.params.lm.softmax.logits_ffn.linear.w":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.m.params.lm.softmax.logits_ffn.linear.w",
                            None, None,
                            just_copy = True),
                    f"opt_states_0_2.v.params.lm.final_ln.scale":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.v.params.lm.final_ln.scale",
                            None, None,
                            just_copy = True),
                    f"opt_states_0_2.v.params.lm.softmax.logits_ffn.linear.w":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.v.params.lm.softmax.logits_ffn.linear.w",
                            None, None,
                            just_copy = True),
            }

        if not self.skip_bias:
            ckpt_map.update({
                    f"opt_states_0_2.m.params.lm.final_ln.bias":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.m.params.lm.final_ln.bias",
                            None, None,
                            just_copy = True),
                    f"opt_states_0_2.v.params.lm.final_ln.bias":
                        self._get_convert_pkg(
                            f"opt_states_0.no_prefix_2.v.params.lm.final_ln.bias",
                            None, None,
                            just_copy = True),
                })

        if not self.skip_position_emb:
            ckpt_map.update({
                f"opt_states_0_2.m.params.lm.position_emb.emb_var":
                    self._get_convert_pkg(
                        f"opt_states_0.no_prefix_2.m.params.lm.position_emb.emb_var",
                        None, None,
                        just_copy = True),
                f"opt_states_0_2.v.params.lm.position_emb.emb_var":
                    self._get_convert_pkg(
                        f"opt_states_0.no_prefix_2.v.params.lm.position_emb.emb_var",
                        None, None,
                        just_copy = True),
            })

        else:
            ckpt_map.update({
                f"opt_states_0_2.m.params.lm.embedding_lookup.emb_var":
                    self._get_convert_pkg(
                        f"opt_states_0.no_prefix_2.m.params.lm.embedding_lookup.emb_var",
                        None, None,
                        just_copy = True),
                f"opt_states_0_2.v.params.lm.embedding_lookup.emb_var":
                    self._get_convert_pkg(
                        f"opt_states_0.no_prefix_2.v.params.lm.embedding_lookup.emb_var",
                        None, None,
                        just_copy = True),
            })


        return ckpt_map
