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
```sh
# If you want to build with updated patches
cd JAX-Toolbox

ROSETTA_BASE=pax

bash .github/container/bump.sh -i .github/container/manifest.yaml
docker buildx build --build-context jax-toolbox=. --tag rosetta-${ROSETTA_BASE}:latest -f rosetta/Dockerfile.${ROSETTA_BASE} --build-arg UPDATE_PATCHES=true .
```

## Development
Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for details on how to extend rosetta.

## Note
This is not an officially supported NVIDIA product
