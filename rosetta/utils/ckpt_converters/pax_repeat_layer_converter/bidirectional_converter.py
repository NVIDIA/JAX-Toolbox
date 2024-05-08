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

import jax
import jax.numpy as jnp
import functools
from common.utils import ConvertHelper

class NonRepeat2RepeatConvertHelper(ConvertHelper):

    @property
    def src_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]
    
    @property
    def target_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']

        return ['mdl_vars.params', f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.m.params", f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.v.params"]

    def create_convert_pkg(self, ckpt_map, base_name: str, num_layers: int, src_shape):
        key = f"lm.transformer.x_layers_0.{base_name}"
        val = self._get_convert_pkg(
            f"lm.transformer.repeat.sub.x_layers_0.{base_name}",
            src_shape, 0,
            extra_src_paths = [f"lm.transformer.x_layers_{i}.{base_name}" for i in range(1, num_layers)],
            stack_dim = 0
        )
        ckpt_map[key] = val
        return ckpt_map

    def _generate_ckpt_map(self):
        num_of_head = self.model_config.num_of_head
        num_gqa_groups = self.model_config.num_gqa_groups
        head_dim = self.model_config.head_dim
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim
        num_layers = self.model_config.num_of_layer

        ckpt_map = {}

        base_names_shapes = [
            ("ff_layer.ffn_layer1.linear.w", (hidden_dim, mlp_intermediate_dim)),
            ("ff_layer.ffn_layer2.linear.w", (mlp_intermediate_dim, hidden_dim)),
            ("ff_layer.layer_norm.scale", (hidden_dim,)),
            ("layer_norm.scale", (hidden_dim,)),
            ("self_attention.post.w", (hidden_dim, num_of_head, head_dim))
        ]

        for name, shape in base_names_shapes:
            ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)
        
        if not self.skip_bias:
            bias_names_shapes = [
                ("ff_layer.ffn_layer1.bias.b", (mlp_intermediate_dim,)),
                ("ff_layer.ffn_layer2.bias.b", (hidden_dim,)),
                ("ff_layer.layer_norm.bias", (hidden_dim,)),
                ("layer_norm.bias", (hidden_dim,)),
                ("self_attention.post.b", (hidden_dim,))
            ]
            
            for name, shape in bias_names_shapes:
                ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)

        if self.use_gated_activations:
            ckpt_map = self.create_convert_pkg(ckpt_map, "ff_layer.ffn_layer1_gate.linear.w", num_layers, (hidden_dim, mlp_intermediate_dim))
 
        if num_gqa_groups:
            qkv_names_shapes = [
                ("self_attention.query.w", (hidden_dim, num_of_head, head_dim)),
                ("self_attention.key.w", (hidden_dim, num_gqa_groups, head_dim)),
                ("self_attention.value.w", (hidden_dim, num_gqa_groups, head_dim))
            ]
            for name, shape in qkv_names_shapes:
                ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)

            if not self.skip_bias:
                qkv_bias_names_shapes = [
                    ("self_attention.query.b", (num_of_head, head_dim)),
                    ("self_attention.key.b", (num_of_head, head_dim)),
                    ("self_attention.value.b", (num_of_head, head_dim))
                ]
                for name, shape in qkv_bias_names_shapes:
                    ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)
        else:
            ckpt_map = self.create_convert_pkg(ckpt_map, "self_attention.combined_qkv.w", num_layers, (3, hidden_dim, num_of_head, head_dim))

            if not self.skip_bias:
                ckpt_map = self.create_convert_pkg(ckpt_map, "self_attention.combined_qkv.b", num_layers, (3, num_of_head, head_dim))

        return ckpt_map


    @property
    def no_prefix_conversions(self):

        num_layers = self.model_config.num_of_layer

        if self.weight_only:
            return {}

        def just_copy(target_name):
            return self._get_convert_pkg(
                target_name,
                None, None,
                just_copy = True
            )
        
        def repeat_count(target_name, num_layers):
            def repeat_fn(arr):
                return jax.numpy.repeat(arr, num_layers)
            return self._get_convert_pkg(
                target_name,
                (), 0,
                repeat_fn,
            )

        ckpt_map = {"opt_states_0_0.count":
                        [just_copy("opt_states_0.no_prefix_0.count"),
                         repeat_count(f"opt_states_0.p#{num_layers}#i-1_0.count", num_layers)],
                    "opt_states_0_1.count":
                        [just_copy(f"opt_states_0.no_prefix_1.count"),
                         repeat_count(f"opt_states_0.p#{num_layers}#i-1_1.count", num_layers)],
                    "opt_states_0_2.count":
                        [just_copy(f"opt_states_0.no_prefix_2.count"),
                         repeat_count(f"opt_states_0.p#{num_layers}#i-1_2.count", num_layers)],
                    "opt_states_0_3.count":
                        [just_copy(f"opt_states_0.no_prefix_3.count"),
                         repeat_count(f"opt_states_0.p#{num_layers}#i-1_3.count", num_layers)],
                    "opt_states_0_2.m.params.lm.final_ln.scale":
                        just_copy("opt_states_0.no_prefix_2.m.params.lm.final_ln.scale"),
                    "opt_states_0_2.m.params.lm.softmax.logits_ffn.linear.w":
                        just_copy("opt_states_0.no_prefix_2.m.params.lm.softmax.logits_ffn.linear.w"),
                    "opt_states_0_2.v.params.lm.final_ln.scale":
                        just_copy("opt_states_0.no_prefix_2.v.params.lm.final_ln.scale"),
                    f"opt_states_0_2.v.params.lm.softmax.logits_ffn.linear.w":
                        just_copy("opt_states_0.no_prefix_2.v.params.lm.softmax.logits_ffn.linear.w"),
            }

        if not self.skip_bias:
            ckpt_map.update({
                    "opt_states_0_2.m.params.lm.final_ln.bias":
                        just_copy("opt_states_0.no_prefix_2.m.params.lm.final_ln.bias"),
                    f"opt_states_0_2.v.params.lm.final_ln.bias":
                        just_copy("opt_states_0.no_prefix_2.v.params.lm.final_ln.bias"),
                })

        if not self.skip_position_emb:
            ckpt_map.update({
                f"opt_states_0_2.m.params.lm.position_emb.emb_var":
                    just_copy("opt_states_0.no_prefix_2.m.params.lm.position_emb.emb_var"),
                f"opt_states_0_2.v.params.lm.position_emb.emb_var":
                    just_copy("opt_states_0.no_prefix_2.v.params.lm.position_emb.emb_var"),
            })

        else:
            ckpt_map.update({
                f"opt_states_0_2.m.params.lm.embedding_lookup.emb_var":
                    just_copy("opt_states_0.no_prefix_2.m.params.lm.embedding_lookup.emb_var"),
                f"opt_states_0_2.v.params.lm.embedding_lookup.emb_var":
                    just_copy("opt_states_0.no_prefix_2.v.params.lm.embedding_lookup.emb_var"),
            })


        return ckpt_map

class Repeat2NonRepeatConvertHelper(ConvertHelper):

    @property
    def src_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']

        return ['mdl_vars.params', f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.m.params", f"opt_states_0.p#{self.model_config.num_of_layer}#i-1_2.v.params"]

    @property
    def target_categories(self):
        if self.weight_only:
            return ['mdl_vars.params']
        return ['mdl_vars.params', "opt_states_0_2.m.params", "opt_states_0_2.v.params"]

    def create_convert_pkg(self, ckpt_map, base_name: str, num_layers: int, src_shape):
        key = f"lm.transformer.repeat.sub.x_layers_0.{base_name}"
        val = self._get_convert_pkg(
            f"lm.transformer.x_layers_0.{base_name}",
            src_shape, 0,
            extra_target_paths = [f"lm.transformer.x_layers_{i}.{base_name}" for i in range(1, num_layers)],
            stack_dim = 0,
        )
        ckpt_map[key] = val
        return ckpt_map

    def _generate_ckpt_map(self):
        num_of_head = self.model_config.num_of_head
        head_dim = self.model_config.head_dim
        num_gqa_groups = self.model_config.num_gqa_groups
        hidden_dim = num_of_head * head_dim
        mlp_intermediate_dim = self.model_config.mlp_intermediate_dim
        num_layers = self.model_config.num_of_layer

        ckpt_map = {}

        base_names_shapes = [
            ("ff_layer.ffn_layer1.linear.w", (num_layers, hidden_dim, mlp_intermediate_dim)),
            ("ff_layer.ffn_layer2.linear.w", (num_layers, mlp_intermediate_dim, hidden_dim)),
            ("ff_layer.layer_norm.scale", (num_layers, hidden_dim,)),
            ("layer_norm.scale", (num_layers, hidden_dim,)),
            ("self_attention.post.w", (num_layers, hidden_dim, num_of_head, head_dim))
        ]

        for name, shape in base_names_shapes:
            ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)

        if not self.skip_bias:
            bias_names_shapes = [
                ("ff_layer.ffn_layer1.bias.b", (num_layers, mlp_intermediate_dim,)),
                ("ff_layer.ffn_layer2.bias.b", (num_layers, hidden_dim,)),
                ("ff_layer.layer_norm.bias", (num_layers, hidden_dim,)),
                ("layer_norm.bias", (num_layers, hidden_dim,)),
                ("self_attention.post.b", (num_layers, hidden_dim,))
            ]

            for name, shape in bias_names_shapes:
                ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)

        if self.use_gated_activations:
            ckpt_map = self.create_convert_pkg(ckpt_map, "ff_layer.ffn_layer1_gate.linear.w", num_layers, (num_layers, hidden_dim, mlp_intermediate_dim))

        if num_gqa_groups:
            qkv_names_shapes = [
                ("self_attention.query.w", (num_layers, hidden_dim, num_of_head, head_dim)),
                ("self_attention.key.w", (num_layers, hidden_dim, num_gqa_groups, head_dim)),
                ("self_attention.value.w", (num_layers, hidden_dim, num_gqa_groups, head_dim))
            ]
            for name, shape in qkv_names_shapes:
                ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)

            if not self.skip_bias:
                qkv_bias_names_shapes = [
                    ("self_attention.query.b", (num_layers, num_of_head, head_dim)),
                    ("self_attention.key.b", (num_layers, num_of_head, head_dim)),
                    ("self_attention.value.b", (num_layers, num_of_head, head_dim))
                ]
                for name, shape in qkv_bias_names_shapes:
                    ckpt_map = self.create_convert_pkg(ckpt_map, name, num_layers, shape)
        else:
            ckpt_map = self.create_convert_pkg(ckpt_map, "self_attention.combined_qkv.w", num_layers, (num_layers, 3, hidden_dim, num_of_head, head_dim))

            if not self.skip_bias:
                ckpt_map = self.create_convert_pkg(ckpt_map, "self_attention.combined_qkv.b", num_layers, (num_layers, 3, num_of_head, head_dim))

        return ckpt_map

    @property
    def no_prefix_conversions(self):

        num_layers = self.model_config.num_of_layer

        if self.weight_only:
            return {}

        def just_copy(target_name):
            return self._get_convert_pkg(
                target_name,
                None, None,
                just_copy = True
            )

        ckpt_map = {"opt_states_0.no_prefix_0.count":
                        just_copy("opt_states_0_0.count"),
                    "opt_states_0.no_prefix_1.count":
                        just_copy("opt_states_0_1.count"),
                    "opt_states_0.no_prefix_2.count":
                        just_copy(f"opt_states_0_2.count"),
                    "opt_states_0.no_prefix_3.count":
                        just_copy(f"opt_states_0_3.count"),
                    "opt_states_0.no_prefix_2.m.params.lm.final_ln.scale":
                        just_copy("opt_states_0_2.m.params.lm.final_ln.scale"),
                    "opt_states_0.no_prefix_2.m.params.lm.softmax.logits_ffn.linear.w":
                        just_copy("opt_states_0_2.m.params.lm.softmax.logits_ffn.linear.w"),
                    "opt_states_0.no_prefix_2.v.params.lm.final_ln.scale":
                        just_copy("opt_states_0_2.v.params.lm.final_ln.scale"),
                    "opt_states_0.no_prefix_2.v.params.lm.softmax.logits_ffn.linear.w":
                        just_copy("opt_states_0_2.v.params.lm.softmax.logits_ffn.linear.w"),
            }

        if not self.skip_bias:
            ckpt_map.update({
                    "opt_states_0.no_prefix_2.m.params.lm.final_ln.bias":
                        just_copy("opt_states_0_2.m.params.lm.final_ln.bias"),
                    f"opt_states_0.no_prefix_2.v.params.lm.final_ln.bias":
                        just_copy("opt_states_0_2.v.params.lm.final_ln.bias"),
                })

        if not self.skip_position_emb:
            ckpt_map.update({
                f"opt_states_0.no_prefix_2.m.params.lm.position_emb.emb_var":
                    just_copy("opt_states_0_2.m.params.lm.position_emb.emb_var"),
                f"opt_states_0.no_prefix_2.v.params.lm.position_emb.emb_var":
                    just_copy("opt_states_0_2.v.params.lm.position_emb.emb_var"),
            })

        else:
            ckpt_map.update({
                f"opt_states_0.no_prefix_2.m.params.lm.embedding_lookup.emb_var":
                    just_copy("opt_states_0_2.m.params.lm.embedding_lookup.emb_var"),
                f"opt_states_0.no_prefix_2.v.params.lm.embedding_lookup.emb_var":
                    just_copy("opt_states_0_2.v.params.lm.embedding_lookup.emb_var"),
            })


        return ckpt_map
