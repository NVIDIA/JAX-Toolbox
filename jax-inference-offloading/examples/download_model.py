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
import argparse
import os
import sys
from contextlib import contextmanager, redirect_stdout


def print_usage():
  print(
    """Download model weights from HF or Kaggle Hub depending on the following
input arguments, and print the local path to stdout:
  --hub: auto/huggingface/hf/kaggle, defaults to auto
  --model: model name, e.g. meta-llama/Llama-3.1-8B-Instruct or google/gemma-3/flax
  --flavor: for kaggle only, e.g. gemma-3-1b-it
  --hf-token: HF token, optional, can also be set via env HF_TOKEN
  --kaggle-username: Kaggle username, optional, can also be set via env KAGGLE_USERNAME
  --kaggle-key: Kaggle key, optional, can also be set via env KAGGLE_KEY
  --ignore: file patterns to ignore, optional, can be a comma-separated list of patterns. Only used by HF.
Example usage:
  python download_model.py --hub=hf --model=meta-llama/Llama-3.1-8B-Instruct --hf-token=xxxx
  python download_model.py --hub=kaggle --model=google/gemma-3/flax --flavor=gemma-3-1b-it --kaggle-username=xxxx --kaggle-key=xxxx
    """.strip()
  )

def parse_args():
  parser = argparse.ArgumentParser(
    description="Download model weights from HF or Kaggle and print the local path.",
  )
  parser.add_argument(
    "--hub",
    default="auto",
    choices=["auto", "huggingface", "hf", "kaggle"],
    help="Which hub to use. Defaults to 'auto'.",
  )
  parser.add_argument(
    "--model",
    required=True,
    help="Model identifier, e.g. 'meta-llama/Llama-3.1-8B-Instruct' or 'google/gemma-3/flax'.",
  )
  parser.add_argument(
    "--flavor",
    help="Kaggle-only flavor, e.g. 'gemma-3-1b-it'.",
  )
  parser.add_argument(
    "--hf-token",
    dest="hf_token",
    default=None,
    help="Hugging Face token. Alternatively set env HF_TOKEN.",
  )
  parser.add_argument(
    "--kaggle-username",
    dest="kaggle_username",
    default=None,
    help="Kaggle username. Alternatively set env KAGGLE_USERNAME.",
  )
  parser.add_argument(
    "--kaggle-key",
    dest="kaggle_key",
    default=None,
    help="Kaggle API key. Alternatively set env KAGGLE_KEY.",
  )
  parser.add_argument(
    "--ignore",
    dest="ignore",
    default=None,
    help="File patterns to ignore (comma-separated). Only used by HF.",
  )
  return parser.parse_args()


def decide_hub(hub, flavor):
  if hub == "auto":
    return "kaggle" if flavor else "hf"
  if hub in ("hf", "huggingface"):
    return "hf"
  if hub == "kaggle":
    return "kaggle"
  return "hf"


def download_hf(model, token, ignore_patterns):
  import huggingface_hub

  return huggingface_hub.snapshot_download(
    repo_id=model,
    token=token,
    ignore_patterns=ignore_patterns,
  )


def download_kaggle(model, flavor, kaggle_username, kaggle_key):
  if kaggle_username:
    os.environ["KAGGLE_USERNAME"] = kaggle_username
  if kaggle_key:
    os.environ["KAGGLE_KEY"] = kaggle_key

  import kagglehub

  model_id = model if not flavor else f"{model}/{flavor}"
  return kagglehub.model_download(model_id)


@contextmanager
def stdout_to_stderr():
  sys.stdout.flush()
  old_stdout_fd = os.dup(1)
  try:
    os.dup2(2, 1)
    with redirect_stdout(sys.stderr):
      yield
  finally:
    sys.stdout.flush()
    os.dup2(old_stdout_fd, 1)
    os.close(old_stdout_fd)


def main():
  if any(a in ("-h", "--help") for a in sys.argv[1:]):
    print_usage()
    sys.exit(0)

  args = parse_args()
  hub = decide_hub(args.hub, args.flavor)

  ignore_patterns = None
  if args.ignore:
    ignore_patterns = [p.strip() for p in args.ignore.split(",") if p.strip()]

  if hub == "hf":
    token = args.hf_token or os.getenv("HF_TOKEN")
    with stdout_to_stderr():
      return download_hf(args.model, token, ignore_patterns)
  elif hub == "kaggle":
    username = args.kaggle_username or os.getenv("KAGGLE_USERNAME")
    key = args.kaggle_key or os.getenv("KAGGLE_KEY")
    with stdout_to_stderr():
      return download_kaggle(args.model, args.flavor, username, key)
  else:
    raise ValueError(f"Unknown hub: {hub}")


if __name__ == "__main__":
  try:
    local_path = main()
    if not local_path:
      raise RuntimeError("Download did not return a local path")
    print(local_path)
  except Exception as err:
    print(f"{err}", file=sys.stderr)
    sys.exit(1)
