BENCHMARK=$1
shift

/scripts/start_ssh.sh ${@};
pushd /scripts;

/scripts/generate_hostfiles.sh ${@};
popd;

BENCHMARK=$BENCHMARK NHOSTS=2 /scripts/test.sh
