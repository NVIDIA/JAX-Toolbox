#!/usr/bin/env bash

set -euo pipefail
# Never trace this script: it handles credentials mounted by BuildKit.
set +x

compiler_cache_args=()
case "${ENABLE_NVTE_SCCACHE:-0}" in
    1)
        required_sccache_secrets=(
            AWS_ACCESS_KEY_ID
            AWS_SECRET_ACCESS_KEY
            AWS_SESSION_TOKEN
            SCCACHE_BUCKET
        )
        for secret_name in "${required_sccache_secrets[@]}"; do
            if [[ ! -s "/run/secrets/${secret_name}" ]]; then
                echo "ENABLE_NVTE_SCCACHE=1 requires the ${secret_name} BuildKit secret" >&2
                exit 1
            fi
        done

        for secret_name in "${required_sccache_secrets[@]}"; do
            printf -v "${secret_name}" '%s' "$(<"/run/secrets/${secret_name}")"
            export "${secret_name}"
        done

        : "${SCCACHE_REGION:?ENABLE_NVTE_SCCACHE=1 requires SCCACHE_REGION}"
        : "${TARGETARCH:?BuildKit did not provide TARGETARCH}"
        export NVTE_CCACHE_BIN=sccache
        export SCCACHE_REGION
        export SCCACHE_S3_KEY_PREFIX="sccache/${TARGETARCH}"
        export SCCACHE_S3_USE_SSL=true
        compiler_cache_args+=(--ccache)
        ;;
    0)
        ;;
    *)
        echo "ENABLE_NVTE_SCCACHE must be 0 or 1" >&2
        exit 1
        ;;
esac

exec build-te.sh "${compiler_cache_args[@]}" "$@"
