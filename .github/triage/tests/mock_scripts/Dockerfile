ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS passing
RUN <<"EOF" bash -exuo pipefail
export DEBIAN_FRONTEND=noninteractive
export TZ=America/Los_Angeles
apt-get update
apt-get install --no-install-recommends --no-install-suggests -y git
apt-get clean
rm -rf /var/lib/apt/lists/*
EOF
ADD build-jax.sh build-te.sh /usr/local/bin
RUN <<"EOF" bash -exuo pipefail
git config --global user.name "Test User"
git config --global user.email "test@example.com"
for package in jax xla; do
  mkdir -p /opt/${package}
  cd /opt/${package}
  git init -b main
  touch pass.txt
  git add pass.txt
  git commit -m C0
  git commit --allow-empty -m C1
done
EOF
WORKDIR /opt/jax
RUN touch feature_file.txt # for the mock build-jax.sh
FROM passing AS failing
RUN git rm pass.txt && git commit -m C2
FROM passing AS passing_with_later_version
RUN git commit --allow-empty -m C2
RUN git commit --allow-empty -m C3
FROM failing AS failing_with_later_version
RUN git commit --allow-empty -m C3
