BENCHMARK=$1
NHOSTS=${NHOSTS:-2}
shift

/scripts/start_ssh.sh ${@};
pushd /scripts;

/scripts/generate_hostfiles.sh ${@};
popd;

if [ $NODE_RANK = 0 ] ; then
  BENCHMARK=$BENCHMARK NHOSTS=$NHOSTS /scripts/test.sh
fi
