# JAX Toolbox

<table>
  <thead>
    <tr>
      <th colspan=4 style="text-align:center;">
        <a href="https://github.com/NVIDIA/JAX-Toolbox/actions/workflows/ci.yaml?query=event%3Aschedule+branch%3Amain">
        <img
          style="height: 1.5em;"
          src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-workflow-metadata.json&logo=github-actions&logoColor=white"
        />
        </a>
      </th>
    </tr>
    <tr>
      <th>Components</th>
      <th>Container</th>
      <th>Build</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=base%3D%7BCUDA%2CcuDNN%2CNCCL%2COFED%2CEFA%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:base</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-base-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-base-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-base-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td></td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=core%3D%7Bbase%2CJAX%2CFlax%2CTE%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:jax</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-jax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-jax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-V100.json&logo=nvidia&label=V100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-jax-unit-test-A100.json&logo=nvidia&label=A100">
        </picture>
        <br>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-unit-test-V100.json&logo=nvidia&label=TE%20V100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-unit-test-A100.json&logo=nvidia&label=TE%20A100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-te-multigpu-test.json&logo=nvidia&label=TE%20Multi%20GPU">
        </picture>
        <br>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-V100.json&logo=nvidia&label=Pallas V100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pallas-unit-test-A100.json&logo=nvidia&label=Pallas A100">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Levanter%3D%7Bcore%2CLevanter%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:levanter</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-levanter-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-levanter-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-unit-test-V100.json&logo=nvidia&label=V100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-levanter-unit-test-A100.json&logo=nvidia&label=A100">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Equinox%3D%7Bcore%2CEquinox%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:equinox</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-equinox-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-equinox-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <!-- <td>
        <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-unit-test-V100.json&logo=nvidia&label=V100">
        <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-equinox-unit-test-A100.json&logo=nvidia&label=A100">
      </td> -->
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Triton%3D%7Bcore%2CJAX-Triton%2CTriton%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:triton</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-triton-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-triton-build-amd64.json&logo=docker&label=amd64"></a>
        <!-- <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-triton-build-arm64.json&logo=docker&label=arm64"> -->
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-triton-unit-test-V100.json&logo=nvidia&label=JAX-Triton V100">
        </picture>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-triton-unit-test-A100.json&logo=nvidia&label=JAX-Triton A100">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Upstream%20T5X%3D%7Bcore%2CT5X%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:upstream-t5x</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-t5x-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-t5x-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-t5x-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-upstream-t5x-mgmn-test.json&logo=nvidia&label=A100%20distributed">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Rosetta%20T5X%3D%7Bcore%2CT5X%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:t5x</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-t5x-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-t5x-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-t5x-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-t5x-mgmn-test.json&logo=nvidia&label=A100%20distributed">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Upstream%20PAX%3D%7Bcore%2Cpaxml%2Cpraxis%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:upstream-pax</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-pax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-upstream-pax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-pax-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-upstream-pax-mgmn-test.json&logo=nvidia&label=A100%20distributed">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Rosetta%20PAX%3D%7Bcore%2Cpaxml%2Cpraxis%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:pax</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-pax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-pax-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-pax-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-build-pax-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-rosetta-pax-mgmn-test.json&logo=nvidia&label=A100%20distributed">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=MaxText%3D%7Bcore%2CMaxText%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:maxtext</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-maxtext-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-build-amd64.json&logo=docker&label=amd64"></a>
        <!-- <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-build-arm64.json&logo=docker&label=arm64"> -->
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-maxtext-test.json&logo=nvidia&label=A100%20distributed">
        </picture>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Grok%3D%7Bcore%2CGrok-1%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:grok</code>
      </td>
      <td>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-grok-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-grok-build-amd64.json&logo=docker&label=amd64"></a>
        <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-grok-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-grok-build-arm64.json&logo=docker&label=arm64"></a>
      </td>
      <td>
      </td>
    </tr>
    <tr>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/static/v1?label=&color=gray&logo=docker&message=Gemma%3D%7Bcore%2CGemma%2CPaliGemma%7D">
        </picture>
      </td>
      <td>
        <code>ghcr.io/nvidia/jax:gemma</code>
      </td>
      <td>
      <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-gemma-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-gemma-build-amd64.json&logo=docker&label=amd64"></a>
      <!-- <a href="https://gist.github.com/nvjax/913c2af68649fe568e9711c2dabb23ae/#file-final-gemma-md"><img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-gemma-build-arm64.json&logo=docker&label=arm64"></a> -->
      </td>
      <td>
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-gemma-unit-test-V100.json&logo=nvidia&label=V100">
        </picture>      
        <picture>
          <img style="height:1em;" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Fnvjax%2F913c2af68649fe568e9711c2dabb23ae%2Fraw%2Fbadge-gemma-unit-test-A100.json&logo=nvidia&label=A100">
        </picture>      
      </td>
    </tr>
  </tbody>
</table>

In all of the above cases, `ghcr.io/nvidia/jax:XXX` points to the most recent
nightly build of the container for `XXX`. These containers are also tagged as
`ghcr.io/nvidia/jax:XXX-YYYY-MM-DD`, if a stable reference is required.

## Note
This repo currently hosts a public CI for JAX on NVIDIA GPUs and covers some JAX libraries like: [T5x](https://github.com/google-research/t5x), [PAXML](https://github.com/google/paxml), [Transformer Engine](https://github.com/NVIDIA/TransformerEngine), [Pallas](https://jax.readthedocs.io/en/latest/pallas/quickstart.html) and others to come soon.

## Supported Models
We currently enable training and evaluation for the following models:
| Model Name | Pretraining | Fine-tuning | Evaluation |
| :--- | :---: | :---: | :---: |
| [GPT-3(paxml)](./rosetta/rosetta/projects/pax) | ✔️ |   | ✔️ |
| [LLaMA2(paxml)](./rosetta/rosetta/projects/pax#llama) |  |   | ✔️ |
| [t5(t5x)](./rosetta/rosetta/projects/t5x) | ✔️ | ✔️ | ✔️ |
| [ViT](./rosetta/rosetta/projects/vit) | ✔️ | ✔️ | ✔️ |
| [Imagen](./rosetta/rosetta/projects/imagen) | ✔️ |   | ✔️ |
| [PaliGemma](./rosetta/rosetta/projects/paligemma) |  | ✔️ | ✔️ |

We will update this table as new models become available, so stay tuned.

## Environment Variables

The [JAX image](https://github.com/NVIDIA/JAX-Toolbox/pkgs/container/jax) is embedded with the following flags and environment variables for performance tuning:

| XLA Flags | Value | Explanation |
| --------- | ----- | ----------- |
| `--xla_gpu_enable_latency_hiding_scheduler` | `true`  | allows XLA to move communication collectives to increase overlap with compute kernels |
| `--xla_gpu_enable_triton_gemm` | `false` | use cuBLAS instead of Trition GeMM kernels |

| Environment Variable | Value | Explanation |
| -------------------- | ----- | ----------- |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | use a single queue for GPU work to lower latency of stream operations; OK since XLA already orders launches |
| `NCCL_NVLS_ENABLE` | `0` | Disables NVLink SHARP ([1](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-nvls-enable)). Future releases will re-enable this feature. |
| `CUDA_MODULE_LOADING` | `EAGER` | Disables lazy-loading ([1](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#cuda-environment-variables)) which uses slightly more GPU memory. |

There are various other XLA flags users can set to improve performance. For a detailed explanation of these flags, please refer to the [GPU performance](./rosetta/docs/GPU_performance.md) doc. XLA flags can be tuned per workflow. For example, each script in [contrib/gpu/scripts_gpu](https://github.com/google/paxml/tree/main/paxml/contrib/gpu/scripts_gpu) sets its own [XLA flags](https://github.com/google/paxml/blob/93fbc8010dca95af59ab615c366d912136b7429c/paxml/contrib/gpu/scripts_gpu/benchmark_gpt_multinode.sh#L30-L33).

## Profiling JAX programs on GPU
See [this page](./docs/profiling.md) for more information about how to profile JAX programs on GPU.

## FAQ (Frequently Asked Questions)

<details>
    <summary>`bus error` when running JAX in a docker container</summary>

**Solution:**
```bash
docker run -it --shm-size=1g ...
```

**Explanation:**
The `bus error` might occur due to the size limitation of `/dev/shm`. You can address this by increasing the shared memory size using
the `--shm-size` option when launching your container.
</details>

<details>

<summary>enroot/pyxis reports error code 404 when importing multi-arch images</summary>

**Problem description:**
```
slurmstepd: error: pyxis:     [INFO] Authentication succeeded
slurmstepd: error: pyxis:     [INFO] Fetching image manifest list
slurmstepd: error: pyxis:     [INFO] Fetching image manifest
slurmstepd: error: pyxis:     [ERROR] URL https://ghcr.io/v2/nvidia/jax/manifests/<TAG> returned error code: 404 Not Found
```

**Solution:**
Upgrade [enroot](https://github.com/NVIDIA/enroot) or [apply a single-file patch](https://github.com/NVIDIA/enroot/releases/tag/v3.4.0) as mentioned in the enroot v3.4.0 release note.

**Explanation:**
Docker has traditionally used Docker Schema V2.2 for multi-arch manifest lists but has switched to using the Open Container Initiative (OCI) format since 20.10. Enroot added support for OCI format in version 3.4.0.
</details>

## JAX on Public Clouds

* AWS
    * [Add EFA integration](https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-efa.html)
    * [SageMaker code sample](https://github.com/aws-samples/aws-samples-for-ray/tree/main/sagemaker/jax_alpa_language_model)
* GCP
    * [Getting started with JAX multi-node applications with NVIDIA GPUs on Google Kubernetes Engine](https://cloud.google.com/blog/products/containers-kubernetes/machine-learning-with-jax-on-kubernetes-with-nvidia-gpus)
* Azure
    * [Accelerating AI applications using the JAX framework on Azure’s NDm A100 v4 Virtual Machines](https://techcommunity.microsoft.com/t5/azure-high-performance-computing/accelerating-ai-applications-using-the-jax-framework-on-azure-s/ba-p/3735314)
* OCI
    * [Running a deep learning workload with JAX on multinode multi-GPU clusters on OCI](https://blogs.oracle.com/cloud-infrastructure/post/running-multinode-jax-clusters-on-oci-gpu-cloud)

## Resources
* [What's New in JAX | GTC Spring 2023](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51956/)
* [Slurm and OpenMPI zero config integration](https://jax.readthedocs.io/en/latest/_autosummary/jax.distributed.initialize.html)
* [Adding custom GPU ops](https://jax.readthedocs.io/en/latest/Custom_Operation_for_GPUs.html)
