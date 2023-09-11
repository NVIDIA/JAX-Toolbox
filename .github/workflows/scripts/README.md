# Github Action Utility Scripts

* "[all_image_tags](./all_image_tags.sh) $GH_TOKEN $IMAGE_REPO"
    * Example: `all_image_tags XXXXXXXXXXXX ghcr.io/nvidia/t5x`
    * Returns all tags on ghcr.io for a given image repo
* "[inspect_remote_img](./inspect_remote_img.sh) $GH_TOKEN $IMAGE"
    * Example: `inspect_remote_img XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest`
    * Returns the metadata of a tagged remote image (no download)
* "[get_build_date](./get_build_date.sh) $GH_TOKEN $IMAGE"
    * Example: `get_build_date XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest`
    * Returns the BUILD_DATE of a tagged remote image (no download)
* "[get_remote_env](./get_remote_env.sh) $GH_TOKEN $IMAGE"
    * Example: `get_remote_env XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest`
    * Returns the ENV of a tagged remote image (no download)
        * Useful to inspect CUDA env vars
* "[get_remote_labels](./get_remote_labels.sh) $GH_TOKEN $IMAGE"
    * Example: `get_remote_labels XXXXXXXXXXXX ghcr.io/nvidia/t5x:latest`
    * Returns the opencontainer annotation labels of a tagged remote image (no download)