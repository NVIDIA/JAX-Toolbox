len() {
  local -r arr=($@)
  echo "${#arr[@]}"
}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

NRANKS_FACTORS=(1 2 4 8)

NHOSTS=$(len "$@")
echo "generating hostfiles for ${NHOSTS} hosts: "
for h in "$@"; do echo "$h"; done

mkdir -p "${SCRIPT_DIR}/hostfiles${NHOSTS}"

for nr in "${NRANKS_FACTORS[@]}";
do
  rm -f "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${nr}"
  touch "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${nr}"
  for h in "$@";
  do
    echo "$h port=22 slots=${nr}" >> "${SCRIPT_DIR}/hostfiles${NHOSTS}/hostfile${nr}"
  done
done
