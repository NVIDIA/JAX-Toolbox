# Base maxtext image
FROM ghcr.io/nvidia/jax:maxtext

RUN apt-get update && \
    apt-get install -y redis && \
    python3 -m pip install redis && \
    python3 -m pip install "ray[default]"
