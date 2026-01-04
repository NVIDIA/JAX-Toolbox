#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from setuptools import Command, setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop
from setuptools.command.sdist import sdist as _sdist


class BuildPackageProtos(Command):
    description = 'build grpc protobuf modules'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from grpc.tools import command
        command.build_package_protos(".", strict_mode=True)


class BuildPy(_build_py):
    def run(self):
        self.run_command('build_protos')
        super().run()


class Sdist(_sdist):
    def run(self):
        self.run_command('build_protos')
        super().run()


class Develop(_develop):  # legacy/compat editable installs only
    def run(self):
        self.run_command('build_protos')
        super().run()

setup(
    name='jax-inference-offloading',
    version='0.0.1',
    packages=['jax_inference_offloading'],
    install_requires=[
        'cupy-cuda12x',
        'cloudpickle',
        'flax',
        'grpcio==1.76.*',
        'protobuf==6.33.*',
        'huggingface-hub',
        'jax==0.8.1',
        'jaxtyping',
        'kagglehub',
        'vllm==0.12.0',
    ],
    extras_require={
        'test': ['pytest>=7.0'],
    },
    cmdclass={
        'build_protos': BuildPackageProtos,
        'build_py': BuildPy,
        'sdist': Sdist,
        'develop': Develop,
    }
)
