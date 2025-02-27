# Base maxtext image
FROM ghcr.io/nvidia/jax:maxtext

RUN mkdir /ray_resiliency_example
WORKDIR /ray_resiliency_example

RUN apt-get update && \
    apt-get install -y redis && \
    python3 -m pip install redis && \
    python3 -m pip install "ray[default]"

COPY ./launch_ray_job.sh /ray_resiliency_example/
COPY ./ray_example.py /ray_resiliency_example/
COPY ./ray_example_driver.py /ray_resiliency_example/
