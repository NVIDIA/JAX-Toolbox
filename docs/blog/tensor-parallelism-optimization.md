# Optimizing Tensor Parallelism in JAX and XLA

> **Draft** — this page is a placeholder. Content is in progress.

## Background

<!-- Why TP matters: model sizes past single-GPU memory, the compute/communication
     tradeoff versus FSDP and pipeline parallelism. -->

## Where tensor parallelism costs performance

<!-- All-gather / reduce-scatter on the critical path, exposed communication,
     small GEMM tiles at high TP degree, NVLink versus inter-node bandwidth. -->

## Optimizations

<!-- One subsection per optimization, with the mechanism and how to enable it. -->

## Results

<!-- Model, hardware, TP degrees swept, throughput and step-time deltas. -->

## Reproducing

<!-- Container tag, framework config flags, launch command. -->

## Takeaways
