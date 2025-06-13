BENCHMARK=$1
NHOSTS=${NHOSTS:-2}
shift

/scripts/start_ssh.sh ${@};
pushd /scripts;

/scripts/generate_hostfiles.sh ${@};
popd;

COMPLETION_FLAG=/opt/output/$BENCHMARK_done

if [ $NODE_RANK = 0 ] ; then
  BENCHMARK=$BENCHMARK NHOSTS=$NHOSTS /scripts/test.sh
  touch $COMPLETION_FLAG
else
  while [ ! -f $COMPLETION_FLAG ]; do sleep 10; done
fi

