PORT=${PORT:-22}

while true; do
  host=$1
  if [[ -z $host ]]; then
    break
  fi
  ssh -p "${PORT}" "$host" \
    echo "Connected to ${host}"
  shift
done
