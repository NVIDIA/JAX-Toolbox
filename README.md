# JAX Toolbox

<table>
    <thead>
        <tr>
            <th>Image</th>
            <th>Build</th>
            <th>Tests</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>

[![container-badge-base]][container-link-base]
            </td>
            <td>
[![build-badge-base]][workflow-base]
            </td>
            <td> n/a </td>
        </tr>
        <tr style="border-bottom-style:hidden">
            <td colspan=3> Frameworks </td>
        </tr>
        <!-- JAX -->
        <tr>
            <td>
[![container-badge-jax]][container-link-jax]  
            </td>
            <td>
[![build-badge-jax]][workflow-jax]
            </td>
            <td>
[![test-badge-jax-V100]][workflow-jax-unit]
<br>
[![test-badge-jax-A100]][workflow-jax-unit]
            </td>
        </tr>
        <!-- t5x -->
        <tr style="border-style:hidden">
            <td>
[![container-badge-t5x]][container-link-t5x]
            </td>
            <td>
[![build-badge-t5x]][workflow-t5x] 
            </td>
            <td>
[![test-badge-t5x]][workflow-t5x-perf] 
            </td>
        </tr>
        <!-- pax -->
        <tr style="border-style:hidden">
            <td>
[![container-badge-pax]][container-link-pax] 
            </td>
            <td>
[![build-badge-pax]][workflow-pax]
            </td>
            <td>
[![test-badge-pax]][workflow-pax-perf]
            </td>
        </tr>
        <!-- te -->
        <tr>
            <td>
[![container-badge-te]][container-link-te]
            </td>
            <td>
[![build-badge-te]][workflow-te]
            </td>
            <td>
[![unit-test-badge-te]][workflow-te-test] <br> [![integration-test-badge-te]][workflow-te-test]
            </td>
        </tr>
        <tr style="border-bottom-style:hidden">
            <td colspan=3> Rosetta </td>
        </tr>
        <!-- rosetta-t5x -->
        <tr style="border-style:hidden">
            <td>
[![container-badge-rosetta-t5x]][container-link-rosetta-t5x] 
            </td>
            <td>
[![build-badge-rosetta-t5x]][workflow-rosetta-t5x] 
            </td>
            <td>
[![test-badge-rosetta-t5x]][workflow-rosetta-t5x]
            </td>
        </tr>
        <!-- rosetta-pax -->
        <tr>
            <td>
[![container-badge-rosetta-pax]][container-link-rosetta-pax]
            </td>
            <td>
[![build-badge-rosetta-pax]][workflow-rosetta-pax]
            </td>
            <td>
[![test-badge-rosetta-pax]][workflow-rosetta-pax]
            </td>
        </tr>
    </tbody>
</table>


[container-badge-base]: https://img.shields.io/static/v1?label=&message=.base&color=gray&logo=docker
[container-badge-jax]: https://img.shields.io/static/v1?label=&message=JAX&color=gray&logo=docker
[container-badge-t5x]: https://img.shields.io/static/v1?label=&message=T5X&color=gray&logo=docker
[container-badge-pax]: https://img.shields.io/static/v1?label=&message=PAX&color=gray&logo=docker
[container-badge-rosetta-t5x]: https://img.shields.io/static/v1?label=&message=ROSETTA(T5X)&color=gray&logo=docker
[container-badge-rosetta-pax]: https://img.shields.io/static/v1?label=&message=ROSETTA(PAX)&color=gray&logo=docker
[container-badge-te]: https://img.shields.io/static/v1?label=&message=TE&color=gray&logo=docker

[container-link-base]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-toolbox
[container-link-jax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax
[container-link-t5x]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/t5x
[container-link-pax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/pax
[container-link-te]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax-te
[container-link-rosetta-t5x]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/rosetta-t5x
[container-link-rosetta-pax]: https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/rosetta-pax

[build-badge-base]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/weekly-base-build.yaml?branch=main&label=weekly&logo=github-actions&logoColor=dddddd
[build-badge-jax]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-jax-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-t5x]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-t5x-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-pax]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-pax-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd
[build-badge-rosetta-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-t5x-build-status.json&logo=github-actions&logoColor=dddddd
[build-badge-rosetta-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-pax-build-status.json&logo=github-actions&logoColor=dddddd
[build-badge-te]: https://img.shields.io/github/actions/workflow/status/NVIDIA/JAX-Toolbox/nightly-te-build.yaml?branch=main&label=nightly&logo=github-actions&logoColor=dddddd

[workflow-base]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/weekly-base-build.yaml
[workflow-jax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-build.yaml
[workflow-t5x]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-t5x-build.yaml
[workflow-pax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pax-build.yaml
[workflow-rosetta-t5x]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-t5x-build-test.yaml
[workflow-rosetta-pax]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-rosetta-pax-build.yaml
[workflow-te]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-te-build.yaml

[test-badge-jax-V100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fjax-unit-test-status-V100.json&logo=nvidia
[test-badge-jax-A100]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fjax-unit-test-status-A100.json&logo=nvidia
[test-badge-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Ft5x-test-completion-status.json&logo=nvidia
[test-badge-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fpax-test-completion-status.json&logo=nvidia
[unit-test-badge-te]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fte-unit-test-status.json&logo=nvidia
[integration-test-badge-te]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fte-integration-test-status.json&logo=nvidia
[test-badge-rosetta-t5x]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-t5x-test-status.json&logo=nvidia
[test-badge-rosetta-pax]: https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Frosetta-pax-test-status.json&logo=nvidia

[workflow-jax-unit]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-jax-test-unit.yaml
[workflow-t5x-perf]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-t5x-test-mgmn.yaml
[workflow-pax-perf]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-pax-test-mgmn.yaml
[workflow-te-test]: https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/nightly-te-test.yaml


## Note
This repo currently hosts a public CI for JAX on NVIDIA GPUs and covers some JAX libraries like: [T5x](https://github.com/google-research/t5x), [PAXML](https://github.com/google/paxml), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), and others to come soon.

## Supported Models
We currently enable training and evaluation for the following models:
| Model Name | Pretraining | Fine-tuning | Evaluation |
| :--- | :---: | :---: | :---: |
| [t5(t5x)](./rosetta/rosetta/projects/t5x) | ✔️ | ✔️ | ✔️ |

We will update this table as new models become available, so stay tuned.
