#!/bin/bash
set -euo pipefail

usage() {
    echo "Install a compiler cache used by build-te.sh"
    echo ""
    echo "  Usage: $0 [ccache|sccache]"
    echo ""
    echo "  If no argument is provided, NVTE_CCACHE_BIN selects the cache."
    exit "${1}"
}

if [[ "$#" -gt 1 ]]; then
    usage 1
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage 0
fi

CACHE_BINARY="${1:-${NVTE_CCACHE_BIN:-ccache}}"
if command -v "${CACHE_BINARY}" &> /dev/null; then
    echo "Compiler cache already installed: $(command -v "${CACHE_BINARY}")"
    exit 0
fi

case "${CACHE_BINARY}" in
    ccache)
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ccache
        rm -rf /var/lib/apt/lists/*
        ;;
    sccache)
        SCCACHE_VERSION="${SCCACHE_VERSION:-v0.17.0}"
        if [[ ! "${SCCACHE_VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
            echo "SCCACHE_VERSION must have the form vX.Y.Z"
            exit 1
        fi

        case "$(dpkg --print-architecture)" in
            amd64)
                SCCACHE_HOST_ARCH=x86_64
                SCCACHE_RUSTUP_HOST=x86_64-unknown-linux-gnu
                ;;
            arm64)
                SCCACHE_HOST_ARCH=aarch64
                SCCACHE_RUSTUP_HOST=aarch64-unknown-linux-gnu
                ;;
            *)
                echo "Unsupported architecture for sccache: $(dpkg --print-architecture)"
                exit 1
                ;;
        esac

        CUDA_TOOLKIT_VERSION="${CUDA_VERSION:-}"
        if command -v nvcc &> /dev/null; then
            DETECTED_CUDA_TOOLKIT_VERSION="$(
                nvcc --version | sed -n 's/.*release \([0-9][0-9.]*\),.*/\1/p'
            )"
            if [[ -n "${DETECTED_CUDA_TOOLKIT_VERSION}" ]]; then
                CUDA_TOOLKIT_VERSION="${DETECTED_CUDA_TOOLKIT_VERSION}"
            fi
        fi

        SCCACHE_TMPDIR="$(mktemp -d)"
        cleanup() {
            if [[ -n "${SCCACHE_TMPDIR:-}" && -d "${SCCACHE_TMPDIR}" ]]; then
                rm -rf -- "${SCCACHE_TMPDIR}"
            fi
        }
        trap cleanup EXIT

        if [[ "${SCCACHE_VERSION}" == "v0.17.0" ]] && \
           [[ -n "${CUDA_TOOLKIT_VERSION}" ]] && \
           dpkg --compare-versions "${CUDA_TOOLKIT_VERSION}" ge "13.3"; then
            # mozilla/sccache#2722, the CUDA 13.3 nvcc dry-run parsing fix,
            # landed after v0.17.0. Build the exact upstream commit containing
            # that fix instead of maintaining and applying a downstream patch.
            SCCACHE_SOURCE_DIR="${SCCACHE_TMPDIR}/sccache"
            SCCACHE_SOURCE_COMMIT="e9b15a35f7240a7edd1b9644583edb388c6cb5f9"
            SCCACHE_RUST_TOOLCHAIN="${SCCACHE_RUST_TOOLCHAIN:-1.88.0}"
            SCCACHE_RUSTUP_VERSION=1.29.0
            SCCACHE_RUSTUP_INIT="${SCCACHE_TMPDIR}/rustup-init"
            SCCACHE_RUSTUP_URL="https://static.rust-lang.org/rustup/archive/${SCCACHE_RUSTUP_VERSION}/${SCCACHE_RUSTUP_HOST}/rustup-init"

            export DEBIAN_FRONTEND=noninteractive
            apt-get update
            apt-get install -y --no-install-recommends libssl-dev pkg-config
            rm -rf /var/lib/apt/lists/*
            git init "${SCCACHE_SOURCE_DIR}"
            git -C "${SCCACHE_SOURCE_DIR}" remote add origin \
                https://github.com/mozilla/sccache.git
            git -C "${SCCACHE_SOURCE_DIR}" fetch --depth 1 \
                origin "${SCCACHE_SOURCE_COMMIT}"
            git -C "${SCCACHE_SOURCE_DIR}" checkout --detach FETCH_HEAD
            if [[ "$(git -C "${SCCACHE_SOURCE_DIR}" rev-parse HEAD)" != \
                  "${SCCACHE_SOURCE_COMMIT}" ]]; then
                echo "Unexpected sccache source commit"
                exit 1
            fi

            export CARGO_HOME="${SCCACHE_TMPDIR}/cargo"
            export RUSTUP_HOME="${SCCACHE_TMPDIR}/rustup"
            wget -nv --tries=5 --retry-connrefused \
                -O "${SCCACHE_RUSTUP_INIT}" "${SCCACHE_RUSTUP_URL}"
            wget -nv --tries=5 --retry-connrefused \
                -O- "${SCCACHE_RUSTUP_URL}.sha256" \
                | awk -v binary="${SCCACHE_RUSTUP_INIT}" \
                      '{print $1"  "binary}' \
                | sha256sum -c -
            chmod 755 "${SCCACHE_RUSTUP_INIT}"
            "${SCCACHE_RUSTUP_INIT}" \
                -y \
                --no-modify-path \
                --profile minimal \
                --default-toolchain "${SCCACHE_RUST_TOOLCHAIN}"
            "${CARGO_HOME}/bin/cargo" build \
                --manifest-path "${SCCACHE_SOURCE_DIR}/Cargo.toml" \
                --release \
                --locked \
                --no-default-features \
                --features=s3
            install -m 755 \
                "${SCCACHE_SOURCE_DIR}/target/release/sccache" \
                /usr/local/bin/sccache
            unset CARGO_HOME RUSTUP_HOME
        else
            SCCACHE_STEM="sccache-${SCCACHE_VERSION}-${SCCACHE_HOST_ARCH}-unknown-linux-musl"
            SCCACHE_URL="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_STEM}.tar.gz"
            SCCACHE_ARCHIVE="${SCCACHE_TMPDIR}/${SCCACHE_STEM}.tar.gz"
            wget -nv --tries=5 --retry-connrefused --waitretry=10 --timeout=60 \
                 --retry-on-http-error=429,500,502,503,504 \
                 -O "${SCCACHE_ARCHIVE}" "${SCCACHE_URL}"
            wget -nv --tries=5 --retry-connrefused -O- "${SCCACHE_URL}.sha256" \
                 | awk -v archive="${SCCACHE_ARCHIVE}" \
                       '{print $1"  "archive}' \
                 | sha256sum -c -
            tar -xzf "${SCCACHE_ARCHIVE}" -C "${SCCACHE_TMPDIR}"
            install -m 755 \
                "${SCCACHE_TMPDIR}/${SCCACHE_STEM}/sccache" \
                /usr/local/bin/sccache
        fi
        ;;
    *)
        echo "${CACHE_BINARY} is not installed; automatic installation supports only ccache and sccache"
        exit 1
        ;;
esac

if ! command -v "${CACHE_BINARY}" &> /dev/null; then
    echo "Compiler cache installation did not provide ${CACHE_BINARY}"
    exit 1
fi
echo "Installed compiler cache: $(command -v "${CACHE_BINARY}")"
