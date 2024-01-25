#!/bin/bash

YELLOW='\033[0;33m'
NOCOLOR='\033[0m'

echo -e "${YELLOW}
WARNING: To speed up the model training it is recommended to cache the datasets with:

    $ python -m levanter.main.cache_dataset --id <dataset_id> --cache_dir <path/to/shared/cache>

When run this container on distributed system, please make sure that the cache folder 
(<path/to/shared/cache>) is on a shared filesystem across nodes.
To supply the cache folder, you can use `--data.cache_dir` flag for Levanter, for example:

    $ docker run -v <path/to/shared/cache>:<path/to/in-docker/cache> <levanter-docker-tag> \\
        python -m levanter.main.train_lm --data.cache_dir <path/to/in-docker/cache>\"

${NOCOLOR}"
