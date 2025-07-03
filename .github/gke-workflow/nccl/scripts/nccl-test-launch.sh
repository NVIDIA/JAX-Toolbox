BENCHMARK=$1
NHOSTS=${NHOSTS:-2}
shift

/scripts/start_ssh.sh ${@};
pushd /scripts;

/scripts/generate_hostfiles.sh ${@};
popd;

COMPLETION_FLAG=/opt/output/${BENCHMARK}_done

service ssh restart

if [ $NODE_RANK = 0 ] ; then
  for host in ${@}; do
    host_ready=false
    while ! $host_ready; do
      status=$(ssh $host echo "ready" 2> /dev/null || echo "unready")
      if [ "$status" = "ready" ]; then
        host_ready=true
        break
      fi
      echo "$host not ready"
      sleep 5
    done
    echo "$host ready"
  done

  NCCL_BENCHMARK=$BENCHMARK NHOSTS=$NHOSTS /scripts/test.sh

  for host in ${@}; do
    ssh ${host} touch ${COMPLETION_FLAG}
  done

else
  while [ ! -f $COMPLETION_FLAG ]; do sleep 10; done
fi

