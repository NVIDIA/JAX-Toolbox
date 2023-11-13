# Rosetta
Project rosetta is a jax project maintained by NVIDIA and allows us to train
LLM, CV, and multimodal models.

## Building rosetta manually

### Building rosetta with a specific base
```bash
ROSETTA_BASE=t5x  # or pax

docker buildx build --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} .

# If you want to specify a specific base image
docker buildx build --tag rosetta:latest -f Dockerfile.${ROSETTA_BASE} --build-arg BASE_IMAGE=ghcr.io/nvidia/${ROSETTA_BASE}:mealkit-YYYY-MM-DD .
```

### Advanced use-cases
```sh
# [T5x Example] If you want to build with a different patchlist (patchlist must be relative to rosetta dir)
docker buildx build --build-arg T5X_PATCHLIST=patches/t5x/patchlist-t5x.txt.gen --build-arg FLAX_PATCHLIST=patches/flax/patchlist-flax.txt.gen --target rosetta --tag rosetta:latest -f Dockerfile.t5x .

# [T5x Example] If you want to build with patches from another image
scripts/extract-patches.sh <img-name>  # Extracts generated patch dir under ./patches/
docker buildx build --build-arg T5X_PATCHLIST=patches/t5x/patchlist-t5x.txt.gen --build-arg FLAX_PATCHLIST=patches/flax/patchlist-flax.txt.gen --target rosetta --tag rosetta:latest -f Dockerfile.t5x .
```

## Development
Please read [DEVELOPMENT.md](docs/DEVELOPMENT.md) for details on how to extend rosetta.

## Note
This is not an officially supported NVIDIA product
