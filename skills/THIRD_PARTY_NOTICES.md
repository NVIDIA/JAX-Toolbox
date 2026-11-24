# Third-Party Notices: agent skills

This file covers every agent skill in this directory (`skills/<skill-name>/`). It does not cover the rest of
JAX-Toolbox (containers, CI and other code).

## Dependencies

Skills do not vendor runtime dependencies. Their scripts and instructions may use the packages and tools below,
which users install separately; none of their code is copied into this directory.

**Table 1. Dependencies: used by skills, installed separately, not included in this directory**

| Project | License | Use |
|---|---|---|
| [Python](https://github.com/python/cpython) standard library | PSF-2.0 | Skill scripts and tests. |
| [JAX and jaxlib](https://github.com/jax-ml/jax) | Apache-2.0 | Imported by skill scripts. jaxlib includes XLA. |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause; bundled components are under 0BSD, MIT, Zlib and CC0-1.0 | Imported by skill scripts. |
| [pip](https://github.com/pypa/pip) | MIT | Run by skill scripts to report installed packages. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | Apache-2.0 | Training framework that skills give instructions for. Users check it out themselves. |
| [XLA](https://github.com/openxla/xla) | Apache-2.0 | The compiler inside jaxlib. Skills refer to its flags. |
| [Transformer Engine](https://github.com/NVIDIA/TransformerEngine) | Apache-2.0 | Library that skills give instructions for. |
| NVIDIA driver (`nvidia-smi`) and [Nsight Systems](https://developer.nvidia.com/nsight-systems) (`nsys`) | NVIDIA proprietary, installed separately | Run as external programs by skill scripts. |

## Third-party material in skill files

Some skill files, mostly evaluation fixtures under `skills/<skill-name>/evals/files/`, contain excerpts of, or
output produced by, third-party projects. The `NOTICE.md` beside such files is the authoritative notice for them:
it gives the source URL (pinned to a commit), the upstream project, the license and the local changes. To list
these notices, run `find skills -name NOTICE.md`.

**Table 2. Third-party material: included in skill files**

| Project | License | Kind of material |
|---|---|---|
| [AI-Hypercomputer/tpu-recipes](https://github.com/AI-Hypercomputer/tpu-recipes) (Copyright Google LLC) | Apache-2.0 | Trimmed excerpts of recipe scripts and READMEs, used as evaluation inputs. |
| [MaxText](https://github.com/AI-Hypercomputer/maxtext) | Apache-2.0 | Training logs from NVIDIA's runs of MaxText, or synthetic logs in its format, that include MaxText's configuration dump; configuration-key lists curated from MaxText's `base.yml`. |
| [XLA](https://github.com/openxla/xla) | Apache-2.0 | Optimized HLO that XLA produced in NVIDIA's runs, and kernel traces of such runs exported with Nsight Systems. |

The Apache-2.0 license text that applies to the material in Table 2 is in the Apache License 2.0 section of the
repository's [LICENSE.md](../LICENSE.md).

## Adding or changing a skill

When a skill adds a dependency, add it to Table 1 with its SPDX license identifier.

When a skill adds a file that copies, adapts or is substantially derived from third-party material:

- add or update the `NOTICE.md` beside that file, with the source URL (pinned to a commit), the upstream project,
  its SPDX license identifier and a summary of the local changes;
- if that license's text is not already in the repository's `LICENSE.md`, include it in the `NOTICE.md`;
- add the project to Table 2 if it is not already listed.
