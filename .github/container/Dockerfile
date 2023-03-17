ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install python-is-python3 python3-pip
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
