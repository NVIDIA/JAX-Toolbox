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

import numpy as np
import logging
import time
import os
import signal
from typing import List, Dict
import pickle as pkl
import copy
import argparse
import functools
import yaml
from yaml import Loader

import uuid
import zmq
import subprocess

from pytriton.model_config import ModelConfig, Tensor, DynamicBatcher
from pytriton.triton import Triton, TritonConfig
from pytriton.decorators import batch

from rosetta.projects.inference_serving.server_utils import pow2list, triton_textencode
from rosetta.projects.inference_serving.shared_numpy import SharedNPDict

# List of strings of comma-separated device indexes. i.e. ['0', '1', '2'] or ['0,1', '2,3']
# Each list element contains the CUDA_VISIBLE_DEVICES visible to an inference process
ModelDevicesType = List[str]

# ZMQ-based infer function. Sends input over socket and reads output return
def infer_fn(socket, **inputs: np.ndarray):
    # start_time = time.time()
    # out = [np.array([pkl.dumps(np.zeros((4096, 128), dtype=np.float32))] * list(inputs.values())[0].shape[0])]
    # out = [np.zeros((list(inputs.values())[0].shape[0], 4096*128), dtype=np.float32)]
    # logging.warning(f'time to create out {time.time() - start_time}')
    # return out
    for k, v in inputs.items():
        #logging.warning(f'inferring on type {v.dtype}')
        inputs[k] = np.array([pkl.dumps(v)])
    start_time = time.time()
    shared_inputs = SharedNPDict(dict_to_share=inputs)
    socket.send_pyobj(shared_inputs.get_metas())
    out = socket.recv_pyobj()
    shared_inputs.close_and_unlink()
    # logging.warning(f'[triton] time to backend {time.time() - start_time}')
    # out_time = time.strftime('%X %x %Z')
    # logging.warning(f'outtime {out_time}')
    if isinstance(out, str):
        return out
    out = SharedNPDict(metadata=out).localize(close_shared=True, unlink_shared=True)#['out']
    #logging.warning(out)
    # return [out['padded_outs'], out['seqlens']]
    return out

def get_infer_fns(device_struct: ModelDevicesType, child_command:str):
    sockets = []
    infer_fns = []
    for dl in device_struct:
        ctx = zmq.Context(io_threads=1)
        socket_addr = "ipc:///tmp/pytriton_multi-" + str(uuid.uuid4())

        socket = ctx.socket(zmq.REQ)
        socket.bind(socket_addr)
        
        sockets.append(socket)

        subprocess.Popen(f'SOCKET_ADDRESS={socket_addr} CUDA_VISIBLE_DEVICES={dl} {child_command} &', shell=True)

    for sock in sockets:
        infer_fns.append(batch(functools.partial(infer_fn, socket=sock)))
        logging.info(f'Built infer_fn for {sock}')

    return infer_fns

def find_model_max_bs(devices: str, server_config:dict, model_name:str):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Finding the maximum batch size for model: {model_name}")

    
    ctx = zmq.Context(io_threads=1)
    socket_addr = "ipc:///tmp/pytriton_MAX_BS_multi-" + str(uuid.uuid4())
    socket = ctx.socket(zmq.REQ)
    socket.bind(socket_addr)
        
    command = server_config['models'][model_name]['run_command']
    proc = subprocess.Popen(f'SOCKET_ADDRESS={socket_addr} CUDA_VISIBLE_DEVICES={devices} {command} &', shell=True, preexec_fn=os.setsid)
    time.sleep(5) #setup time

    lower = 1 
    test = 256 
    upper_fail = None

    while upper_fail is None or ((float(upper_fail) / lower >= 1.125) and upper_fail - lower > 1):
        logging.info(f'Trying bs {test}')
        socket.send_pyobj({'singleton': test})
        out = socket.recv_pyobj()

        if isinstance(out, str):
            # logging.warning(f'bs: {test} failed with error: {out}')
            if test == 1:
                logging.info('bs of 1 has failed. Exiting')
                exit()
            else:
                upper_fail = test
                test = (lower + upper_fail) // 2
        else:
            logging.info(f'bs: {test} succeeded')
            if upper_fail is None:
                lower = test
                test *= 2
            else:
                lower = test
                test = (lower + upper_fail) // 2
                
        logging.info(f'New lower: {lower}, test: {test}, upper: {upper_fail}')

    socket.close()
    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    return lower

def get_batchsize(device_struct:ModelDevicesType, gpu_name:str, server_config:dict, model_name:str):
    devices = device_struct[0]
    if gpu_name is None:
        logging.info("No gpu_name given. Finding max_bs")
        return find_model_max_bs(devices, server_config, model_name)

    for config_gpu, max_bs in server_config['models'][model_name]['max_bs'].items():
        if gpu_name in config_gpu:
            logging.info(f"Matched gpu name: {gpu_name} to configuration {config_gpu} with max_bs {max_bs}")
            if max_bs is None:
                logging.info("Since found max_bs is None, finding max_bs")
                max_bs = find_model_max_bs(devices, server_config, model_name)
            return max_bs

    logging.info(f'GPU name {gpu_name} not found in config. Finding max_bs')
    return find_model_max_bs(devices, server_config, model_name)

def config_to_tensor(triton_in_out_config):
    out = []
    for inp in triton_in_out_config:
        first_key = list(inp.keys())[0]
        out += [
             Tensor(name=first_key, 
                    dtype=np.dtype(inp[first_key]['dtype']), 
                    shape=inp[first_key]['shape']) 
            ]
    return out

def triton_run(port:int, device_structs:Dict[str, ModelDevicesType], gpu_name:str, server_config:dict):
    logging.warning(f'port {port}, devices {device_structs}')
    triton_config = TritonConfig(http_port=port, grpc_port=port+1000, metrics_port=port+2000, log_verbose=0)
    with Triton(config=triton_config) as triton:
        for model_name in server_config['models'].keys():
            model_cfg = server_config['models'][model_name]
            logging.warning(f'Setting up model {model_name} with configuration: {model_cfg}')
            batch_size = get_batchsize(device_structs[model_name], gpu_name, server_config, model_name)   
            logging.warning(f'Using batch size {batch_size}')
        
            infer_fns = get_infer_fns(device_structs[model_name], model_cfg['run_command'])

            dyn_batch = DynamicBatcher(
                max_queue_delay_microseconds = 100000, 
                preferred_batch_size = pow2list(batch_size)
                )

            triton.bind(
                model_name=model_name,
                infer_func=infer_fns,
                inputs=config_to_tensor(model_cfg['inputs']), 
                outputs=config_to_tensor(model_cfg['outputs']),
                config=ModelConfig(max_batch_size=batch_size, batching=True, batcher=dyn_batch),
            )
        triton.serve()

def build_visible_device_structs(devices_available, 
                                 total_devices, 
                                 total_device_first_idx,
                                 server_config) -> Dict[str, ModelDevicesType]:
    def single_model_devices(model_name, devices_available):
        device_list = []
        devices_per_process = server_config['models'][model_name]['gpus_per_process']

        # Number of devices must be divisible by the number of devices per process
        assert len(devices_available) % devices_per_process == 0

        for proc_id in range(len(devices_available) // devices_per_process):
            device_list.append(','.join(devices_available[proc_id * devices_per_process: (proc_id + 1) * devices_per_process]))

        return device_list

    if len(server_config['models'].keys()) > 1:
        assert total_devices is not None, "total_devices must be given if using more than one model"
        assert total_device_first_idx is not None, "total_device_first_idx must be given if using more than one model"

    # Figuring out gpus per model
    device_ctr = 0
    device_counts: dict[str, int] = {}
    gpus_per_model_proc = {}
    for model_name in server_config['models'].keys():
        model_cfg_resources = server_config['models'][model_name]['resources']
        gpus_per_model_proc[model_name] = server_config['models'][model_name]['gpus_per_process']
        if model_cfg_resources['fraction'] is not None:
            count = round(total_devices * model_cfg_resources['fraction'])
        elif model_cfg_resources['count'] is not None:
            count = model_cfg_resources['count']
        else:
            assert False, f'No resources specified for {model_name}'

        count = count // gpus_per_model_proc[model_name] * gpus_per_model_proc[model_name]
        device_ctr += count
        device_counts[model_name] = count

    model_names = list(device_counts.keys())
    if device_ctr > total_devices:
        logging.info(f"Current resource specification is using too many devices! ({device_counts}; \
                      Available: {total_devices} This can be due to rounding errors with fractional resources \
                      or too many devices specified under 'count'. Reducing devices until program can run")

        idx = 0
        since_last_update = 0
        while device_counts > total_devices:
            model_name = model_names[idx]
            if device_counts[model_name] > gpus_per_model_proc[model_name]:
                device_counts[model_name] -= gpus_per_model_proc[model_name]
                device_ctr -= gpus_per_model_proc[model_name]
                since_last_update = 0
            else:
                since_last_update += 1
            idx += 1
            idx %= len(model_names)
            if since_last_update > len(model_names):
                assert False, "There are not enough devices to run 1 process of each model"

    if device_ctr < total_devices:
        logging.info(f'Warning, {total_devices - device_ctr} devices idle')

    logging.info(f'Device arrangement: {device_counts}')

    before_this_host = total_device_first_idx
    while before_this_host > 0:
        for model in model_names:
            if device_counts[model] > 0:
                device_counts[model] -= gpus_per_model_proc[model]
                before_this_host -= gpus_per_model_proc[model]
                break

    logging.warning(f'gpus_per_process {gpus_per_model_proc}, device_counts {device_counts}, devices_available {devices_available}')
    devices_per_model: Dict[str, List[int]] = {}
    remaining_devices = copy.deepcopy(devices_available)
    while len(remaining_devices) > 0:
        for model in model_names:
            logging.warning(f'remaining_devices {remaining_devices}, gpus_per_model_proc {gpus_per_model_proc}')
            if device_counts[model] > 0:
                assert gpus_per_model_proc[model] <= len(remaining_devices), "Cannot evenly fit model onto this host"
                if model not in devices_per_model.keys():
                    devices_per_model[model] = []
                devices_per_model[model] += remaining_devices[:gpus_per_model_proc[model]]
                remaining_devices = remaining_devices[gpus_per_model_proc[model]:]
                break

    # final construction
    visible_device_structs = {}
    for model, devices in devices_per_model.items():
        visible_device_structs[model] = single_model_devices(model, devices)

    return visible_device_structs


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='PyTriton Inference server with many GPUs communicating over zmq')
  parser.add_argument(
      '--port',
      type=int,
      default=1234,
      help="port for server")
  parser.add_argument(
      '--devices',
      type=str,
      required=True,
      help="Comma-separated list of GPU indexes available")
  parser.add_argument(
      '--total_devices',
      type=int,
      required=False,
      help="Total number of inferencing devices. This is required when using multiple models.")
  parser.add_argument(
      '--total_device_first_idx',
      type=int,
      required=False,
      help="Index of first device out of all inference devices. I.e. if this is the second 8-gpu host \
            doing inference, then this argument should be 8. Required for multiple model inference")
  parser.add_argument(
      '--gpu_name',
      type=str,
      required=False,
      help="Used to match up to batch size configs. This program will check if any keys under 'max_bs' \
            are substrings of this name and use the first hit. I.e. a100_80g. If no default is set, \
            it will run the max batch size finder")
  parser.add_argument(
    '--config_file',
    type=str,
    required=True,
    help='YAML configuration for this server')

  args = parser.parse_args()

  with open(args.config_file, 'r') as f:
      server_config = yaml.load(f.read(), Loader=Loader)

  # Figure out devices I can use
  all_devices = args.devices.split(',')
  visible_device_structs = \
          build_visible_device_structs(all_devices, args.total_devices, \
                                       args.total_device_first_idx, server_config)

  triton_run(args.port, visible_device_structs, args.gpu_name, server_config)

