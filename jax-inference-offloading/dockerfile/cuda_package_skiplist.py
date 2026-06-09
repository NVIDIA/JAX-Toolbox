#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filter CUDA wheel payload packages that are already provided by the base image."""

from __future__ import annotations

import argparse
import csv
import email.message
import fnmatch
import os
import pathlib
import re
import sys
from dataclasses import dataclass


CUDA_LIB_DIRS = (
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/lib",
    "/usr/local/cuda/targets/sbsa-linux/lib",
    "/usr/local/cuda/targets/x86_64-linux/lib",
)


# Package skiplist: these Python wheel packages only carry CUDA runtime payloads
# that the CUDA DL base image already owns under /usr/local/cuda.
PACKAGE_SKIPLIST = {
    "nvidia-cublas": ("libcublas.so*",),
    "nvidia-cublas-cu13": ("libcublas.so*",),
    "nvidia-cuda-cupti": ("libcupti.so*",),
    "nvidia-cuda-cupti-cu13": ("libcupti.so*",),
    "nvidia-cuda-nvrtc": ("libnvrtc.so*",),
    "nvidia-cuda-nvrtc-cu13": ("libnvrtc.so*",),
    "nvidia-cuda-runtime": ("libcudart.so.13*",),
    "nvidia-cuda-runtime-cu13": ("libcudart.so.13*",),
    "nvidia-cudnn": ("libcudnn.so*",),
    "nvidia-cudnn-cu13": ("libcudnn.so*",),
    "nvidia-cufft": ("libcufft.so*",),
    "nvidia-cufft-cu13": ("libcufft.so*",),
    "nvidia-cufile": ("libcufile.so*",),
    "nvidia-cufile-cu13": ("libcufile.so*",),
    "nvidia-curand": ("libcurand.so*",),
    "nvidia-curand-cu13": ("libcurand.so*",),
    "nvidia-cusolver": ("libcusolver.so*",),
    "nvidia-cusolver-cu13": ("libcusolver.so*",),
    "nvidia-cusparse": ("libcusparse.so*",),
    "nvidia-cusparse-cu13": ("libcusparse.so*",),
    "nvidia-cusparselt": ("libcusparseLt.so*",),
    "nvidia-cusparselt-cu13": ("libcusparseLt.so*",),
    "nvidia-nccl": ("libnccl.so*",),
    "nvidia-nccl-cu13": ("libnccl.so*",),
    "nvidia-nvjitlink": ("libnvJitLink.so*",),
    "nvidia-nvjitlink-cu13": ("libnvJitLink.so*",),
    "nvidia-nvtx": ("libnvToolsExt.so*",),
    "nvidia-nvtx-cu13": ("libnvToolsExt.so*",),
}

REQ_LINE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^;\\\s]+)(.*)$")


@dataclass(frozen=True)
class SkippedPackage:
    name: str
    version: str
    requirement: str


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def dist_info_name(name: str) -> str:
    return normalize_name(name).replace("-", "_")


def find_library(pattern: str, lib_dirs: tuple[str, ...]) -> pathlib.Path | None:
    for lib_dir in lib_dirs:
        root = pathlib.Path(lib_dir)
        if not root.exists():
            continue
        for path in root.iterdir():
            if fnmatch.fnmatch(path.name, pattern):
                return path
    return None


def has_base_libraries(package_name: str, lib_dirs: tuple[str, ...]) -> bool:
    return all(
        find_library(pattern, lib_dirs) is not None
        for pattern in PACKAGE_SKIPLIST[normalize_name(package_name)]
    )


def filter_requirements(input_path: pathlib.Path, output_path: pathlib.Path, skipped_path: pathlib.Path,
                        lib_dirs: tuple[str, ...]) -> None:
    skipped: list[SkippedPackage] = []
    output_lines: list[str] = []

    for line in input_path.read_text().splitlines(keepends=True):
        match = REQ_LINE_RE.match(line.strip())
        if match and normalize_name(match.group(1)) in PACKAGE_SKIPLIST:
            package = SkippedPackage(
                name=match.group(1),
                version=match.group(2),
                requirement=line.strip(),
            )
            if has_base_libraries(package.name, lib_dirs):
                skipped.append(package)
                output_lines.append(f"# skipped CUDA wheel payload: {line}")
            else:
                output_lines.append(line)
            continue
        output_lines.append(line)

    output_path.write_text("".join(output_lines))
    with skipped_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("name", "version", "requirement"))
        for package in skipped:
            writer.writerow((package.name, package.version, package.requirement))


def marker_metadata(package: SkippedPackage) -> str:
    message = email.message.Message()
    message["Metadata-Version"] = "2.1"
    message["Name"] = package.name
    message["Version"] = package.version
    message["Summary"] = "CUDA runtime payload supplied by the base image"
    return str(message)


def write_markers(skipped_path: pathlib.Path, site_packages: pathlib.Path) -> None:
    with skipped_path.open(newline="") as file:
        packages = [
            SkippedPackage(name=row["name"], version=row["version"], requirement=row["requirement"])
            for row in csv.DictReader(file)
        ]

    for package in packages:
        dist_info = site_packages / f"{dist_info_name(package.name)}-{package.version}.dist-info"
        dist_info.mkdir(parents=True, exist_ok=True)
        (dist_info / "METADATA").write_text(marker_metadata(package))
        (dist_info / "INSTALLER").write_text("cuda-package-skiplist\n")
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: cuda-package-skiplist\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        (dist_info / "RECORD").write_text("")


def default_site_packages() -> pathlib.Path:
    import site

    candidates = site.getsitepackages()
    if not candidates:
        raise SystemExit("Could not determine site-packages path")
    return pathlib.Path(candidates[0])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter")
    filter_parser.add_argument("--input", type=pathlib.Path, required=True)
    filter_parser.add_argument("--output", type=pathlib.Path, required=True)
    filter_parser.add_argument("--skipped", type=pathlib.Path, required=True)
    filter_parser.add_argument(
        "--lib-dir",
        action="append",
        default=list(CUDA_LIB_DIRS),
        help="CUDA library directory to inspect; may be provided more than once",
    )

    mark_parser = subparsers.add_parser("mark-installed")
    mark_parser.add_argument("--skipped", type=pathlib.Path, required=True)
    mark_parser.add_argument("--site-packages", type=pathlib.Path, default=None)

    args = parser.parse_args(argv)
    if args.command == "filter":
        filter_requirements(args.input, args.output, args.skipped, tuple(args.lib_dir))
    elif args.command == "mark-installed":
        write_markers(args.skipped, args.site_packages or default_site_packages())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
