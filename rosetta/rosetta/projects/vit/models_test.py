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

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
import pytest

from rosetta.projects.vit import config, models
from rosetta.projects.vit.layers import FlaxGViTForImageClassificationModule
from t5x import optimizers, utils


def small_vit_module(dtype):

    conf = config.GoogleViTConfig(
        hidden_size = 384,
        num_hidden_layers = 12,
        num_attention_heads = 6,
        intermediate_size = 1536,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
        patch_size = 16,
        encoder_stride = 16,
        classifier = 'token',
        ## single linear layer for fine-tuning
        representation_size = None,
        num_classes = 1000,
        dtype = dtype,
    )
    return FlaxGViTForImageClassificationModule(conf)


def timeit(fn: Callable, num_iter=1, warmup_iter=1):
    """Calculates avg wall clock time"""
    for _ in range(warmup_iter):
        fn()
    start = time.time()
    for _ in range(num_iter):
        fn()
    elapsed = time.time() - start
    return elapsed / num_iter


@pytest.mark.manual
@pytest.mark.perf
def test_vit_bfloat16_speedup(rng):

    vit_module_f32 = small_vit_module(dtype=jnp.float32)

    vit_module_bf16 = small_vit_module(dtype=jnp.bfloat16)

    vit_model_f32 = models.ViTModel(module=vit_module_f32, optimizer_def=None)

    vit_model_bf16 = models.ViTModel(module=vit_module_bf16, optimizer_def=None)

    state = vit_model_bf16.get_initial_variables(
        rng,
        input_shapes={
            'images': (32, 224, 224, 3),
        },
        input_types={
            'images': jnp.float32,
        },
    )
    params = state['params']

    batch = {
        'images': jnp.ones((32, 224, 224, 3), jnp.float32),
        'labels': jnp.zeros((32, 1000), dtype=jnp.float32),
    }

    times = {}
    baseline_jit = None
    for model_name in ('vit_model_f32', 'vit_model_bf16'):
        header = f'{model_name}'
        model = locals()[model_name]

        jitted_call = jax.jit(model.loss_fn)
        jit_time = timeit(lambda: jitted_call(params, batch, rng)[0].block_until_ready())
        times[f'(jit) {header}'] = jit_time

        if model_name == 'vit_model_f32':
            baseline_jit = jit_time

    print('======================')
    print('==== FWD SUMMARY =====')
    print('======================')
    max_fwd_speedup = float('-inf')
    for name, ttime in sorted(times.items(), key=lambda x: x[1]):
        speedup = baseline_jit / ttime
        max_fwd_speedup = max(max_fwd_speedup, speedup)
        print(f'{ttime*1000:7.3f}ms ({speedup:3.1f}x) {name}')

    assert max_fwd_speedup > 1.6

    ## dummy lr schedule
    schedule = utils.create_learning_rate_scheduler(
        factors='linear_decay',
        base_learning_rate=0.0,
        warmup_steps=100,
        min_learning_rate=0.00001,
        decay_factor=1e-6,
    )

    optimizer = optax.adamw(
        learning_rate=schedule, weight_decay=0.02, b1=0.9, b2=0.999, eps=1e-8,
    )

    OPTIMIZER = optimizers.chain(transformations=[optax.clip_by_global_norm(max_norm=1.0), optimizer])

    step_times = {}
    step_baseline_jit = None
    for model_name in ('vit_model_f32', 'vit_model_bf16'):
        header = f'{model_name}'
        model = locals()[model_name]

        # optax stuff
        optax_state = optimizer.init(params)

        def loss_fn(params, batch, rng):
            return model.loss_fn(params, batch, rng)[0]

        def one_step(optax_state, params, batch, rng):
            grads = jax.grad(loss_fn)(params, batch, rng)
            updates, optax_state = optimizer.update(grads, optax_state, params)
            params = optax.apply_updates(params, updates)
            return params, updates

        ###

        jitted_call = jax.jit(one_step)
        jit_time = timeit(
            lambda: jax.block_until_ready(jitted_call(optax_state, params, batch, rng)[0]),
        )
        step_times[f'(jit) {header}'] = jit_time

        if model_name == 'vit_model_f32':
            step_baseline_jit = jit_time

    print('=======================')
    print('==== STEP SUMMARY =====')
    print('=======================')
    max_step_speedup = float('-inf')
    for name, ttime in sorted(step_times.items(), key=lambda x: x[1]):
        speedup = step_baseline_jit / ttime
        max_step_speedup = max(max_step_speedup, speedup)
        print(f'{ttime*1000:7.3f}ms ({speedup:3.1f}x) {name}')

    assert max_step_speedup > 1.6
