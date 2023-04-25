ARG BASE_IMAGE=ghcr.io/nvidia/t5x:latest
FROM ${BASE_IMAGE} AS distribution-builder

COPY . /opt/rosetta
WORKDIR /opt/rosetta
RUN bash create-t5x-distribution.sh -p patchlist.txt -d /opt/t5x
# TODO(terry): revisit whether any of these are needed
RUN rm -rf .git $(find /opt -name "__pycache__") #&& git submodule deinit -f t5x
  
FROM ${BASE_IMAGE} AS rosetta
COPY --from=distribution-builder /opt/t5x /opt/t5x
COPY --from=distribution-builder /opt/rosetta /opt/rosetta

WORKDIR /opt/rosetta
RUN pip install -e /opt/rosetta && rm -rf ~/.cache/pip/

FROM ${BASE_IMAGE} AS rosetta-devel
COPY --from=distribution-builder /opt/t5x /opt/t5x
COPY --from=distribution-builder /opt/rosetta /opt/rosetta

WORKDIR /opt/rosetta
RUN pip install -e '/opt/rosetta[test,lint]' && rm -rf ~/.cache/pip/

