#!/bin/bash
set -euo pipefail

# TODO double check this, as it may be very clunky
if [ "$#" -ne 15 ]; then
  echo "Usage: $0 IMAGE LOG_FILE OUTPUT_PATH TIME_LIMIT NODES GPUS_PER_NODE NTASKS NTASKS_PER_NODE EXTRA_EXPORTS SRUN_PREAMBLE SRUN_SCRIPT SLURM_LOGIN_USER SLURM_LOGIN_HOSTNAME CONTAINER_REGISTRY_TOKEN OUTPUT_MOUNTPOINT" >&2
  exit 1
fi

IMAGE="$1"
LOG_FILE="$2"
OUTPUT_PATH="$3"
TIME_LIMIT="$4"
NODES="$5"
GPUS_PER_NODE="$6"
NTASKS="$7"
NTASKS_PER_NODE="$8"
EXTRA_EXPORTS="$9"
SRUN_PREAMBLE="${10}"
SRUN_SCRIPT="${11}"
SLURM_LOGIN_USER="${12}"
SLURM_LOGIN_HOSTNAME="${13}"
CONTAINER_REGISTRY_TOKEN="${14}"
OUTPUT_MOUNTPOINT="${15}"

SSH_CMD="ssh ${SLURM_LOGIN_USER}@${SLURM_LOGIN_HOSTNAME}"

# Create output directory
${SSH_CMD} "mkdir -p ${OUTPUT_PATH}"

# SLURM submission script
JOB_SCRIPT=$(cat <<'EOF'
#!/bin/bash
#SBATCH --job-name=${GITHUB_RUN_ID}-${INPUT_NAME}
#SBATCH --exclusive
#SBATCH --nodes=__NODES__
#SBATCH --gpus-per-node=__GPUS_PER_NODE__
#SBATCH --time=__TIME_LIMIT__
#SBATCH --output=__LOG_FILE__
#SBATCH --export="__EXTRA_EXPORTS__,ENROOT_PASSWORD=__CONTAINER_REGISTRY_TOKEN__"

# Execute this function when `scancel` is sent
cleanup() {
    echo "Checking for running Docker container..."
    running_container=$(docker ps -q --filter "name=^/__IMAGE__$")

    if [ -n "$running_container" ]; then
        echo "Stopping Docker container __IMAGE__ ..."
        docker stop "__IMAGE__"
    else
        echo "No running container found with name __IMAGE__."
    fi   
}
# Make trap: this will be executed when the scirpt receives a SIGTERM signal
trap 'cleanup' TERM

# Preload enroot container using one task per node
time srun \
  --ntasks-per-node=1 \
  --container-name=runtime \
  --container-image=__IMAGE__ \
  true

# run single-task preambles for, e.g., dependencies installation
time srun \
  --ntasks-per-node=1 \
  --container-name=runtime \
  bash -c '__SRUN_PREAMBLE__'

# Run main tasks with shared container per node
time srun \
  --tasks=__NTASKS__ \
  --tasks-per-node=__NTASKS_PER_NODE__ \
  --container-name=runtime \
  --container-mounts=__OUTPUT_PATH__:__OUTPUT_MOUNTPOINT__ \
  --container-entrypoint bash -c '__SRUN_SCRIPT__'
EOF
)

# Replace placeholders with the actual values
JOB_SCRIPT="${JOB_SCRIPT//__NODES__/${NODES}}"
JOB_SCRIPT="${JOB_SCRIPT//__GPUS_PER_NODE__/${GPUS_PER_NODE}}"
JOB_SCRIPT="${JOB_SCRIPT//__TIME_LIMIT__/${TIME_LIMIT}}"
JOB_SCRIPT="${JOB_SCRIPT//__LOG_FILE__/${LOG_FILE}}"
JOB_SCRIPT="${JOB_SCRIPT//__EXTRA_EXPORTS__/${EXTRA_EXPORTS}}"
JOB_SCRIPT="${JOB_SCRIPT//__CONTAINER_REGISTRY_TOKEN__/${CONTAINER_REGISTRY_TOKEN}}"
JOB_SCRIPT="${JOB_SCRIPT//__IMAGE__/${IMAGE}}"
JOB_SCRIPT="${JOB_SCRIPT//__SRUN_PREAMBLE__/${SRUN_PREAMBLE}}"
JOB_SCRIPT="${JOB_SCRIPT//__NTASKS__/${NTASKS}}"
JOB_SCRIPT="${JOB_SCRIPT//__NTASKS_PER_NODE__/${NTASKS_PER_NODE}}"
JOB_SCRIPT="${JOB_SCRIPT//__OUTPUT_PATH__/${OUTPUT_PATH}}"
JOB_SCRIPT="${JOB_SCRIPT//__OUTPUT_MOUNTPOINT__/${OUTPUT_MOUNTPOINT}}"

# Pass an input name if defined in the environment otherwise default
export INPUT_NAME="${NAME:-ci_job}"

# Submit the job via SSH and capture the job ID
SLURM_JOB_ID=$(${SSH_CMD} "sbatch --parsable" <<EOF
${JOB_SCRIPT}
EOF
)

echo "SLURM_JOB_ID=${SLURM_JOB_ID}" >> "$GITHUB_OUTPUT"
