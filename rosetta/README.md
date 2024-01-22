# Rosetta
Project rosetta is a jax project maintained by NVIDIA and allows us to train
LLM, CV, and multimodal models.

## Building rosetta manually

### Building rosetta with a specific base
```bash
cd JAX-Toolbox

ROSETTA_BASE=pax  # or t5x

docker buildx build --build-context jax-toolbox=. --tag rosetta-${ROSETTA_BASE}:latest -f rosetta/Dockerfile.${ROSETTA_BASE} .

# If you want to specify a specific base image
docker buildx build --build-context jax-toolbox=. --tag rosetta-${ROSETTA_BASE}:latest -f rosetta/Dockerfile.${ROSETTA_BASE} --build-arg BASE_IMAGE=ghcr.io/nvidia/upstream-${ROSETTA_BASE}:mealkit-YYYY-MM-DD .
```

### Advanced use-cases

#### Build with updated patches
```sh
bash .github/container/bump.sh -i .github/container/manifest.yaml
docker buildx build --build-context jax-toolbox=. --tag rosetta-${ROSETTA_BASE}:latest -f rosetta/Dockerfile.${ROSETTA_BASE} --build-arg UPDATE_PATCHES=true .
```

#### Build and force update TE
Supports any git-ref on [NVIDIA/TransformerEngine](https://github.com/NVIDIA/TransformerEngine) including pull requests (e.g., `pull/PR_NUM/head`)
```sh
docker buildx build --build-arg UPDATED_TE_REF=pull/609/head --build-context jax-toolbox=. --tag rosetta-${ROSETTA_BASE}:latest -f rosetta/Dockerfile.${ROSETTA_BASE} .
```

## Development
Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for details on how to extend rosetta.

## Note
This is not an officially supported NVIDIA product
