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
case "${CACHE_BINARY}" in
    ccache)
        if command -v ccache &> /dev/null; then
            echo "Compiler cache already installed: $(command -v ccache)"
            exit 0
        fi
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends ccache
        rm -rf /var/lib/apt/lists/*
        ;;
    sccache)
        readonly SCCACHE_SOURCE_URL="https://github.com/mozilla/sccache.git"
        readonly SCCACHE_SOURCE_COMMIT="e9b15a35f7240a7edd1b9644583edb388c6cb5f9"
        readonly SCCACHE_RUST_TOOLCHAIN="1.88.0"
        readonly SCCACHE_RUSTUP_VERSION="1.29.0"

        case "$(dpkg --print-architecture)" in
            amd64)
                SCCACHE_RUSTUP_HOST="x86_64-unknown-linux-gnu"
                ;;
            arm64)
                SCCACHE_RUSTUP_HOST="aarch64-unknown-linux-gnu"
                ;;
            *)
                echo "Unsupported architecture for sccache: $(dpkg --print-architecture)"
                exit 1
                ;;
        esac

        SCCACHE_TMPDIR="$(mktemp -d)"
        cleanup() {
            rm -rf -- "${SCCACHE_TMPDIR}"
        }
        trap cleanup EXIT

        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends libssl-dev pkg-config
        rm -rf /var/lib/apt/lists/*

        export CARGO_HOME="${SCCACHE_TMPDIR}/cargo"
        export RUSTUP_HOME="${SCCACHE_TMPDIR}/rustup"
        SCCACHE_RUSTUP_INIT="${SCCACHE_TMPDIR}/rustup-init"
        SCCACHE_RUSTUP_URL="https://static.rust-lang.org/rustup/archive/${SCCACHE_RUSTUP_VERSION}/${SCCACHE_RUSTUP_HOST}/rustup-init"

        wget -nv --tries=5 --retry-connrefused \
            -O "${SCCACHE_RUSTUP_INIT}" "${SCCACHE_RUSTUP_URL}"
        wget -nv --tries=5 --retry-connrefused \
            -O- "${SCCACHE_RUSTUP_URL}.sha256" \
            | awk -v binary="${SCCACHE_RUSTUP_INIT}" '{print $1"  "binary}' \
            | sha256sum -c -
        chmod 755 "${SCCACHE_RUSTUP_INIT}"
        "${SCCACHE_RUSTUP_INIT}" \
            -y \
            --no-modify-path \
            --profile minimal \
            --default-toolchain "${SCCACHE_RUST_TOOLCHAIN}"

        "${CARGO_HOME}/bin/cargo" install sccache \
            --git "${SCCACHE_SOURCE_URL}" \
            --rev "${SCCACHE_SOURCE_COMMIT}" \
            --locked \
            --no-default-features \
            --features=s3 \
            --bin sccache \
            --root /usr/local \
            --no-track \
            --force
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
