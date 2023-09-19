# The commit pining only work for XLA and JAX.
set -e

ARCH=`uname -m`

REPO_JAX="https://github.com/google/jax.git"
REPO_XLA="https://github.com/openxla/xla.git"
T5X_REPO="${T5X_REPO:-https://github.com/google-research/t5x.git}"
FLAX_REPO="${FLAX_REPO:-https://github.com/google/flax.git}"
TE_REPO="${TE_REPO:-https://github.com/NVIDIA/TransformerEngine.git}"
PAXML_REPO="${PAXML_REPO:-https://github.com/google/paxml.git}"
PRAXIS_REPO="${PRAXIS_REPO:-https://github.com/google/praxis.git}"

if [ `false` ]; then
    pushd /tmp
    mkdir tmp_clone
    cd tmp_clone
    git clone --depth 1 $REPO_JAX
    cd jax
    REF_JAX=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $REPO_XLA
    cd xla
    REF_XLA=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $T5X_REPO
    cd t5x
    REF_T5X=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $FLAX_REPO
    cd flax
    REF_FLAX=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $TE_REPO
    cd TransformerEngine
    REF_TE=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $PAXML_REPO
    cd paxml
    REF_PAXML=`git rev-parse HEAD`
    cd ..
    git clone --depth 1 $PRAXIS_REPO
    cd praxis
    REF_PRAXIS=`git rev-parse HEAD`
    cd ..
    popd
    rm -rf /tmp/tmp_clone

    echo REF_JAX=$REF_JAX
    echo REF_XLA=$REF_XLA
    echo REF_T5X=$REF_T5X
    echo REF_FLAX=$REF_FLAX
    echo REF_TE=$REF_TE
    echo REF_PAXML=$REF_PAXML
    echo REF_PRAXIS=$REF_PRAXIS
    exit 0
else
    # 2023-09-13
    REF_JAX=5a15ba90db1054753cac368fe6cfe1428b0f05b7
    REF_XLA=7ce02bfdadfc0f67b6c1784a6ada7d3344f8e7af
    REF_T5X=ee8e2782b89b28bc208be2c719900e686edfb5f1
    REF_FLAX=ca3ea06f78834137dfb49dc6c1a0c26fb962003a
    REF_TE=a150d286a76f6cd4e34b0e16e8b9cc11c4dba34c
    REF_PAXML=aa7e69b0bf1cb683f58415dd2b658b634bb78516
    REF_PRAXIS=f062966203536ffa05ee82393333dc1d0e8eb943
    REF_PRAXIS=f062966203536ffa05ee82393333dc1d0e8eb943
fi



#install-t5x.sh -r REF
#install-flax.sh -r REF
#install-te.sh -r
#install-pax.sh --ref_paxml=REF --ref_praxis=REF

docker build --network=host -f Dockerfile.base -t base .
DOCKER_BUILDKIT=1 docker build --network=host -f Dockerfile.jax -t jax --build-arg BASE_IMAGE=base --build-arg REF_JAX=$REF_JAX --build-arg REF_XLA=$REF_XLA .
# the REF doesn't work for T5X and TE!
DOCKER_BUILDKIT=1 docker build --network=host -f Dockerfile.t5x -t t5x --build-arg BASE_IMAGE=jax --build-arg T5X_REF=$REF_T5X --build-arg REF_TE=$REF_TE .
# TODO: Missing REF*
DOCKER_BUILDKIT=1 docker build --network=host -f Dockerfile.aarch64.pax -t pax --build-arg BASE_IMAGE=jax .

#docker tag jax gitlab-master.nvidia.com:5005/fbastien/scripts:jax_gh_manual_cuda12.2_2023-09-13_$ARCH
#docker tag pax gitlab-master.nvidia.com:5005/fbastien/scripts:pax_gh_manual_cuda12.2_2023-09-13_$ARCH
#docker tag t5x gitlab-master.nvidia.com:5005/fbastien/scripts:t5x_gh_manual_cuda12.2_2023-09-13_$ARCH
#docker push gitlab-master.nvidia.com:5005/fbastien/scripts:jax_gh_manual_cuda12.2_2023-09-13_$ARCH
#docker push gitlab-master.nvidia.com:5005/fbastien/scripts:pax_gh_manual_cuda12.2_2023-09-13_$ARCH
#docker push gitlab-master.nvidia.com:5005/fbastien/scripts:t5x_gh_manual_cuda12.2_2023-09-13_$ARCH
