# Copyright (c) 2022-2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import subprocess
import os
import fcntl
import time

parser = argparse.ArgumentParser(description='Run either training or inference server based on process ID')

parser.add_argument('--proc_id', type=int, required=False, default=-1)
parser.add_argument('--proc_total_ct', type=int, required=True)
parser.add_argument('--inf_server_ct', type=int, required=True)
parser.add_argument('--gpus_per_node', type=int, default=-1, required=False) #Needed for multinode
parser.add_argument('--gpu_collection_size', type=int, default=1, required=False)

parser.add_argument('--train_run_command', type=str, required=True)
parser.add_argument('--inf_server_run_command', type=str, required=True)
parser.add_argument('--hostnames_file', type=str, required=True)

parser.add_argument('--inf_log_file', type=str, required=False)

args = parser.parse_args()

train_servers = args.proc_total_ct - args.inf_server_ct

PROCESS_ID = args.proc_id if args.proc_id >= 0 else None

if PROCESS_ID is None and os.getenv('SLURM_PROCID') is not None:
    PROCESS_ID = int(os.getenv('SLURM_PROCID'))
    if PROCESS_ID is None:
        raise ValueError("Failed to get process ID when specializing")

gpus_in_device = args.gpus_per_node if args.gpus_per_node > 0 else args.proc_total_ct
local_id = PROCESS_ID % gpus_in_device
# one inference server per node case
if PROCESS_ID == train_servers or (PROCESS_ID >= train_servers and local_id == 0):
    inf_id = PROCESS_ID - train_servers
    hostname = subprocess.check_output(["hostname", "-I"])
    hostname = hostname.split()[0].decode("utf-8") 
    port = 2345 + inf_id
    hostname = hostname + ':' + str(port) + '\n'

    devices = [str(i) for i in range(local_id, gpus_in_device)]
    device_str = ','.join(devices)

    with open(args.hostnames_file, 'a') as hf:
        # Should be under buffer size on hostname, preventing 
        # writing race conditions, but will lock to be safe.
        fcntl.flock(hf, fcntl.LOCK_EX)
        hf.write(hostname) 
        fcntl.flock(hf, fcntl.LOCK_UN)

    inf_command_withargs = args.inf_server_run_command + f' --port={port} --devices={device_str} --total_device_first_idx={inf_id}'
    # if args.inf_log_file is not None:
        # inf_command_withport += f' &> {args.inf_log_file}'

    print("Inference Command: " + inf_command_withargs)
    if args.inf_log_file is not None:
        with open(args.inf_log_file, 'w') as f:
            subprocess.call(inf_command_withargs, stdout=f, stderr=f, shell=True)
    else:
        os.system(inf_command_withargs)


# train server case
elif PROCESS_ID < train_servers:
    time.sleep(10)
    os.system(f'PROC_ID={PROCESS_ID} ' + args.train_run_command)
