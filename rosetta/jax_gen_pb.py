import os
import sys
from etils import epath
import jax
from jax.experimental import profiler as exp_profiler


profile_dir = epath.Path(sys.argv[1])
directories = profile_dir.glob('plugins/profile/*/')
directories = [d for d in directories if d.is_dir()]
rundir = directories[-1]
print(f'[jax_gen_pb] Jax profile rundir: {rundir}')

# Post process the profile
fdo_profile = exp_profiler.get_profiled_instructions_proto(os.fspath(rundir))

# Save the profile proto to a file.
dump_dir = epath.Path(f"{sys.argv[2]}/{sys.argv[3]}.pb")
dump_dir.parent.mkdir(parents=True, exist_ok=True)
dump_dir.write_bytes(fdo_profile)
