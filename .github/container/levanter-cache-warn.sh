#!/bin/bash

YELLOW='\033[0;33m'
NOCOLOR='\033[0m'

echo -e "${YELLOW}
WARNING: To speed up the model training it is recommended to cache the datasets
inside docker container with:

$ python -m levanter.main.cache_dataset --id <dataset_id> --cache_dir cache/

If you have already done this, please feel free to disregard the message.
${NOCOLOR}"
