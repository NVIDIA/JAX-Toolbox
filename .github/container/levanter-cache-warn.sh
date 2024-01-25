#!/bin/bash

YELLOW='\033[0;33m'
NOCOLOR='\033[0m'

echo -e "${YELLOW}
WARNING: Levanter uses a cache folder to store preprocessed datasets. The position of
the cache default to ~/.cache but can be changed via the `--data.cache_dir` flag:

    $ python -m levanter.main.train_lm ... --data.cache_dir <path/to/cache>

When running on a distributed system, please make sure that the cache folder is placed on a network-readable path,
e.g. a path in an object store (e.g. GCS). or a path to a pre-built cache., that is accessible by all participating nodes.

    $ docker run -v <path/to/shared/filesystem>:<path/to/cache> <levanter-docker-image> ...
    OR
    $ srun --container-mounts=<path/to/shared/filesystem>:<path/to/cache> --container-image=<levanter-docker-image> ...

${NOCOLOR}"
