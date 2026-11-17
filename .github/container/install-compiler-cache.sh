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
        # v0.18.0 includes the CUDA 13.3 nvcc dry-run parsing fix from
        # mozilla/sccache#2722, so an unreleased source commit is no longer needed.
        readonly SCCACHE_VERSION="v0.18.0"

        case "$(dpkg --print-architecture)" in
            amd64)
                readonly SCCACHE_HOST_ARCH="x86_64"
                readonly SCCACHE_ARCHIVE_SHA256="45f1447fbe231e3037bde351ef70677dd212216c8d62ae7ca409fecc4d6acc89"
                ;;
            arm64)
                readonly SCCACHE_HOST_ARCH="aarch64"
                readonly SCCACHE_ARCHIVE_SHA256="2b3284d5da3b46a47dc4229e75bb7b88ac4aa99c8d754fb7d2f84997e5a4354a"
                ;;
            *)
                echo "Unsupported architecture for sccache: $(dpkg --print-architecture)"
                exit 1
                ;;
        esac

        readonly SCCACHE_STEM="sccache-${SCCACHE_VERSION}-${SCCACHE_HOST_ARCH}-unknown-linux-musl"
        readonly SCCACHE_URL="https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_STEM}.tar.gz"
        SCCACHE_TMPDIR="$(mktemp -d)"
        readonly SCCACHE_ARCHIVE="${SCCACHE_TMPDIR}/${SCCACHE_STEM}.tar.gz"
        cleanup() {
            rm -rf -- "${SCCACHE_TMPDIR}"
        }
        trap cleanup EXIT

        wget -nv --tries=5 --retry-connrefused \
            --waitretry=10 --timeout=60 \
            --retry-on-http-error=429,500,502,503,504 \
            -O "${SCCACHE_ARCHIVE}" "${SCCACHE_URL}"
        printf '%s  %s\n' "${SCCACHE_ARCHIVE_SHA256}" "${SCCACHE_ARCHIVE}" \
            | sha256sum -c -
        tar -xzf "${SCCACHE_ARCHIVE}" -C "${SCCACHE_TMPDIR}"
        install -m 755 \
            "${SCCACHE_TMPDIR}/${SCCACHE_STEM}/sccache" \
            /usr/local/bin/sccache
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
