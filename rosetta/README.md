# Rosetta
Project rosetta is a jax project maintained by NVIDIA and allows us to train
LLM, CV, and multimodal models.

## Building rosetta manually

### Building rosetta with a specific base
```bash
ROSETTA_BASE=t5x  # or pax

docker buildx build --target rosetta --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} .

# If you want a devel image with test dependencies
docker buildx build --target rosetta-devel --tag rosetta-devel:latest -f Dockerfile.${ROSETTA_BASE} .

# If you want to specify a specific base image
docker buildx build --target rosetta --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} --build-arg BASE_IMAGE=ghcr.io/nvidia/${ROSETTA_BASE}:nightly-2023-05-01 .
```

## Development
Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for details on how to extend rosetta.

## Note
This is not an officially supported NVIDIA product
