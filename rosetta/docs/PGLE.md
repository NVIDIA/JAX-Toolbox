## PGLE Workflow for GPU with NSight profiler
This doc will describe the steps to run with PGLE (Profile Guided Latency Estimator) support.

On high-level, the **Profile Guided Latency Estimator (PGLE)** workflow measures the actual running time of compute and collectives. This profile information is then fed back into XLA compiler for a better scheduling decision. The first run is basically a *"Profile"* run and then the second run is basically the *"Performance"* run.

The workflow to use the Profile Guided Latency Estimator with Nsight Systems (Nsys) in XLA/GPU is:

1. **Profile Run**: We need to run the workload once, with the async collectives and latency hiding scheduler disabled. To be specific we need to turn the following three XLA flags off. All the remaining flags should be unchanged.

```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=false
--xla_gpu_enable_async_all_gather=false
--xla_gpu_enable_async_reduce_scatter=false"
```
The main reason to do this is to not have any overlaps so that we can get exact costs for different ops.

2. **Generate protobuf**: Once we have the nsys profile generated, we pass it to the python script provided [here [generate_pbtxt.py]](https://github.com/abhinavgoel95/jax/blob/patch-1/jax/tools/pgo_nsys_converter.py) to generate the pbtxt file. A sample pbtxt file would look like this:
```
...
costs { name: "all-gather-start.1" cost_us: 7040.5215 }
costs { name: "all-gather-start.10" cost_us: 4988.425 }
costs { name: "all-gather-start.11" cost_us: 6757.2605 }
...
...
costs { name: "loop_convert_fusion.111" cost_us: 69.215 }
costs { name: "custom-call.848.0" cost_us: 1066.7755 }
costs { name: "custom-call.849.0" cost_us: 1068.728 }
costs { name: "cublas-gemm.209.0" cost_us: 969.08 }
...
...
costs { name: "loop_broadcast_fusion" cost_us: 1.328 }
costs { name: "reduce-scatter.289.1" cost_us: 8609.677 }
costs { name: "reduce-scatter.295" cost_us: 12205.4735 }
costs { name: "wrapped_transpose" cost_us: 2.192 }
costs { name: "wrapped_transpose.1" cost_us: 2.2875 }
...
``` 
It can have many entries and it is the expected behaviour. One important thing to mention is that, this protobuf file is dependent on the model, XLA version, XLA flags and individual run settings. If any of these get changed, the profiling needs to be run again.

3. **Performance Run**: Finally we have the performance run and this time we make sure we have set the above 3 XLA flags to be true. In addition to that we also provide another XLA flag with the pbtxt file.

```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_async_all_gather=true
--xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_pgle_profile_file_or_directory_path=path\to\generated\pbtxt"
```
With that we should see good overlap between the computations and collectives.

Optionally, we can set the following environment variables to ensure that the latency hiding scheduler is indeed using the profiling costs to schedule:
```
export TF_CPP_VMODULE=profile_guided_latency_estimator=10
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=100
```
This should print something like this:
```
PGLE found async wrapped instruction: custom-call.4.1 in custom-call-start.1
PGLE found latency for async op custom-call-start.1 and (assumed)custom-call-done.1 in instruction costs
```

### Recommended XLA Flags

In order to get the best performance with PGLE, here is a list of all recommended XLA flags:
```
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
--xla_gpu_enable_async_all_gather=true
--xla_gpu_enable_async_reduce_scatter=true
--xla_gpu_enable_triton_gemm=false
--xla_gpu_simplify_all_fp_conversions
--xla_gpu_graph_level=0
--xla_gpu_enable_async_all_reduce=true
--xla_gpu_enable_highest_priority_async_stream=true
--xla_gpu_all_reduce_combine_threshold_bytes=1073741824
--xla_gpu_all_gather_combine_threshold_bytes=1073741824
--xla_gpu_reduce_scatter_combine_threshold_bytes=1073741824
--xla_gpu_enable_pipelined_all_gather=true
--xla_gpu_enable_pipelined_reduce_scatter=true
--xla_gpu_enable_pipelined_all_reduce=true
--xla_gpu_enable_while_loop_double_buffering=true
--xla_gpu_enable_triton_softmax_fusion=false
--xla_gpu_enable_all_gather_combine_by_dim=false
--xla_gpu_enable_reduce_scatter_combine_by_dim=false
--xla_disable_hlo_passes=rematerialization
--xla_gpu_pgle_profile_file_or_directory_path=path\to\generated\pbtxt"
```
### About Combine thresholds

One last thing to add is regarding the **"combine thresholds"**. Ideally, a higher combining threshold for all-gather, reduce-scatter kernels will ensure the best use of the bandwidth. However, they might also incur some dependencies as XLA would try to combine several async collectives and we might see degradation in overall overlap. On the other hand, using very small combining thresholds would create more number of individual collective ops. This makes overlap easier but might under-utilize the bandwidth. For the best performance, these values may need to be tuned based on the individual model and number of devices used for training.