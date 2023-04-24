# Rosetta
Project rosetta is a jax project maintained by NVIDIA and allows us to train
LLM, CV, and multimodal models.

## Installing within jax-toolbox/t5x
```bash
cd rosetta

# Init submodules
#  > git submodule update --init --recursive
# Or update
#  > git submodule update --recursive --remote

docker buildx build --target rosetta --tag rosetta:latest .
# Or if you want a devel image with test dependencies
docker buildx build --target rosetta-devel --tag rosetta-devel:latest .
```

## Development
Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for details on how to extend rosetta.

## Note
This is not an officially supported NVIDIA product
