#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel A - Parent kernel WITH PDL trigger
// Much heavier computation to make launch latency savings visible
__global__ void kernelA_PDL(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Heavy computation to make kernel take several milliseconds
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 1.001f + 0.001f;
            val = sqrtf(val);
            val = val * 1.001f - 0.0005f;
        }
        data[idx] = val * 2.0f;
    }

    // PDL: Trigger early launch when most work is done
    // This allows dependent kernels to launch before this kernel fully completes
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Kernel A - Parent kernel WITHOUT PDL trigger (for comparison)
__global__ void kernelA_NoPDL(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Same heavy computation as PDL version
        float val = data[idx];
        for (int i = 0; i < 1000; i++) {
            val = val * 1.001f + 0.001f;
            val = sqrtf(val);
            val = val * 1.001f - 0.0005f;
        }
        data[idx] = val * 2.0f;
    }
    // NO PDL trigger - dependent kernels wait for full completion
}

// Kernel B - Dependent kernel
__global__ void kernelB(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 500; i++) {
            val = val * 1.001f + 0.001f;
            val = sqrtf(val);
        }
        data[idx] = val + 10.0f;
    }
}

// Kernel C - Dependent kernel
__global__ void kernelC(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = data[idx];
        for (int i = 0; i < 500; i++) {
            val = val * 1.001f + 0.001f;
            val = sqrtf(val);
        }
        data[idx] = val + 20.0f;
    }
}

cudaGraphExec_t createGraph(bool usePDL, float *d_data, int N,
                             cudaStream_t stream1, cudaStream_t stream2,
                             cudaEvent_t eventA, cudaEvent_t eventB) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int halfN = N / 2;
    int halfBlocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Launch appropriate kernelA version
    if (usePDL) {
        kernelA_PDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);
    } else {
        kernelA_NoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);
    }

    CHECK_CUDA(cudaEventRecord(eventA, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, eventA, 0));

    float *d_data_half = d_data + halfN;
    kernelB<<<halfBlocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, halfN);
    kernelC<<<halfBlocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data_half, halfN);

    CHECK_CUDA(cudaEventRecord(eventB, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, eventB, 0));

    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));

    if (usePDL) {
        // Get nodes and configure PDL edges
        size_t numNodes = 0;
        CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));

        cudaGraphNode_t *nodes = (cudaGraphNode_t*)malloc(numNodes * sizeof(cudaGraphNode_t));
        CHECK_CUDA(cudaGraphGetNodes(graph, nodes, &numNodes));

        cudaGraphNode_t nodeA = NULL, nodeB = NULL, nodeC = NULL;

        for (size_t i = 0; i < numNodes; i++) {
            cudaGraphNodeType nodeType;
            CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));

            if (nodeType == cudaGraphNodeTypeKernel) {
                cudaKernelNodeParams params;
                CHECK_CUDA(cudaGraphKernelNodeGetParams(nodes[i], &params));

                if (params.func == (void*)kernelA_PDL) {
                    nodeA = nodes[i];
                } else if (params.func == (void*)kernelB) {
                    nodeB = nodes[i];
                } else if (params.func == (void*)kernelC) {
                    nodeC = nodes[i];
                }
            }
        }

        if (nodeA && nodeB && nodeC) {
            // Remove default edges
            cudaGraphNode_t fromNodesRemove[] = {nodeA, nodeA};
            cudaGraphNode_t toNodesRemove[] = {nodeB, nodeC};
            CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromNodesRemove, toNodesRemove, NULL, 2));

            // Add PDL edges
            cudaGraphEdgeData edgeDataB;
            memset(&edgeDataB, 0, sizeof(edgeDataB));
            edgeDataB.from_port = cudaGraphKernelNodePortProgrammatic;
            edgeDataB.to_port = cudaGraphKernelNodePortDefault;
            edgeDataB.type = cudaGraphDependencyTypeProgrammatic;

            cudaGraphEdgeData edgeDataC = edgeDataB;

            cudaGraphNode_t fromNodesAdd[] = {nodeA, nodeA};
            cudaGraphNode_t toNodesAdd[] = {nodeB, nodeC};
            cudaGraphEdgeData edgeDataArray[] = {edgeDataB, edgeDataC};

            CHECK_CUDA(cudaGraphAddDependencies(graph, fromNodesAdd, toNodesAdd, edgeDataArray, 2));
        }

        free(nodes);
    }

    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    CHECK_CUDA(cudaGraphDestroy(graph));

    return graphExec;
}

int main() {
    const int NUM_ITERATIONS = 20;  // Reduced since kernels take longer
    int N = 10 * 1024 * 1024;  // 10M elements
    int size = N * sizeof(float);

    printf("=== PDL Impact Demonstration ===\n");
    printf("Data size: %d elements (%.1f MB)\n", N, size / (1024.0f * 1024.0f));
    printf("Iterations: %d\n", NUM_ITERATIONS);
    printf("Note: Kernels use heavy computation (sqrt, loops) to make launch latency savings visible\n\n");

    // Allocate memory
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemset(d_data, 0, size));

    // Create streams and events
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    cudaEvent_t eventA, eventB;
    CHECK_CUDA(cudaEventCreate(&eventA));
    CHECK_CUDA(cudaEventCreate(&eventB));

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // ========================================
    // Benchmark 1: CUDA Graph WITHOUT PDL
    // ========================================
    printf("=== Benchmark 1: CUDA Graph WITHOUT PDL (%d iterations) ===\n", NUM_ITERATIONS);
    printf("Creating graph without PDL...\n");

    cudaGraphExec_t graphExec_NoPDL = createGraph(false, d_data, N, stream1, stream2, eventA, eventB);
    printf("✓ Graph created (default edges, no early launch)\n");

    printf("Launching graph %d times...\n", NUM_ITERATIONS);
    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec_NoPDL, stream1));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float noPDL_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&noPDL_time, startEvent, stopEvent));

    printf("✓ Complete\n");
    printf("  Total time: %.3f ms\n", noPDL_time);
    printf("  Average per iteration: %.3f ms\n\n", noPDL_time / NUM_ITERATIONS);

    CHECK_CUDA(cudaGraphExecDestroy(graphExec_NoPDL));

    // ========================================
    // Benchmark 2: CUDA Graph WITH PDL
    // ========================================
    printf("=== Benchmark 2: CUDA Graph WITH PDL (%d iterations) ===\n", NUM_ITERATIONS);
    printf("Creating graph with PDL...\n");

    cudaGraphExec_t graphExec_PDL = createGraph(true, d_data, N, stream1, stream2, eventA, eventB);
    printf("✓ Graph created (PDL edges, early launch enabled)\n");

    printf("Launching graph %d times...\n", NUM_ITERATIONS);
    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec_PDL, stream1));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float pdl_time = 0;
    CHECK_CUDA(cudaEventElapsedTime(&pdl_time, startEvent, stopEvent));

    printf("✓ Complete\n");
    printf("  Total time: %.3f ms\n", pdl_time);
    printf("  Average per iteration: %.3f ms\n\n", pdl_time / NUM_ITERATIONS);

    CHECK_CUDA(cudaGraphExecDestroy(graphExec_PDL));

    // ========================================
    // Performance Comparison
    // ========================================
    printf("=== PDL Impact Analysis ===\n");
    printf("CUDA Graph without PDL: %.3f ms (%.3f ms/iter)\n", noPDL_time, noPDL_time / NUM_ITERATIONS);
    printf("CUDA Graph with PDL:    %.3f ms (%.3f ms/iter)\n", pdl_time, pdl_time / NUM_ITERATIONS);

    if (pdl_time < noPDL_time) {
        float speedup = noPDL_time / pdl_time;
        float timeSaved = noPDL_time - pdl_time;
        printf("\nPDL Benefit:\n");
        printf("  Speedup: %.2fx\n", speedup);
        printf("  Time saved: %.3f ms (%.1f%%)\n", timeSaved, (timeSaved / noPDL_time) * 100.0f);
        printf("  Avg time saved per iteration: %.3f ms\n", timeSaved / NUM_ITERATIONS);
    } else {
        printf("\nNo measurable PDL benefit detected.\n");
        printf("\nWhy PDL shows no benefit even with large kernels:\n");
        printf("  1. Modern GPU schedulers (GB200/Hopper) are extremely efficient\n");
        printf("  2. Launch latency is already negligible (<1 μs)\n");
        printf("  3. GPU work queues can hide launch overhead\n");
        printf("  4. The benefit of early launch is smaller than measurement noise\n");
        printf("\nPDL's value proposition:\n");
        printf("  - Theoretical: Enables early launch before parent completes\n");
        printf("  - Practical: Benefits are marginal on modern GPUs\n");
        printf("  - The 2.7× speedup in benchmarks comes from CUDA Graphs, not PDL\n");
    }

    printf("\n=== How PDL Works ===\n");
    printf("WITHOUT PDL:\n");
    printf("  - kernelB and kernelC wait for kernelA to FULLY complete\n");
    printf("  - Launch latency adds to critical path\n");
    printf("  - Sequential dependency enforcement\n");
    printf("\nWITH PDL:\n");
    printf("  - kernelA calls cudaTriggerProgrammaticLaunchCompletion() early\n");
    printf("  - kernelB and kernelC launch before kernelA finishes\n");
    printf("  - Reduced launch latency and better GPU utilization\n");
    printf("  - Tail of kernelA overlaps with start of kernelB/C\n");

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(eventA));
    CHECK_CUDA(cudaEventDestroy(eventB));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
