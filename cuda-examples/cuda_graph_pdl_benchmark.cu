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

// Kernel A - Parent kernel that uses Programmatic Dependent Launch (PDL)
__global__ void kernelA(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }

    // PDL: Signal dependent kernels to launch early
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x - 1) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Kernel B - Dependent kernel that processes first half of data
__global__ void kernelB(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 10.0f;
    }
}

// Kernel C - Dependent kernel that processes second half of data
__global__ void kernelC(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + 20.0f;
    }
}

int main() {
    const int NUM_ITERATIONS = 1000;
    int N = 1024;
    int size = N * sizeof(float);

    printf("=== CUDA Graph with PDL - Performance Comparison ===\n");
    printf("Iterations: %d\n\n", NUM_ITERATIONS);

    // Allocate host memory
    float *h_data = (float*)malloc(size);
    float *h_data_backup = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
        h_data_backup[i] = (float)i;
    }

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));

    // Create 2 streams for parallel execution
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Create events for synchronization
    cudaEvent_t eventA, eventB;
    CHECK_CUDA(cudaEventCreate(&eventA));
    CHECK_CUDA(cudaEventCreate(&eventB));

    // Create timing events
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int halfN = N / 2;
    int halfBlocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    // ========================================
    // Benchmark 1: Traditional Kernel Launches
    // ========================================
    printf("=== Benchmark 1: Traditional Kernel Launches (%d iterations) ===\n", NUM_ITERATIONS);

    CHECK_CUDA(cudaMemcpy(d_data, h_data_backup, size, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Launch kernelA on stream1
        kernelA<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);

        // Record event after kernelA for cross-stream synchronization
        CHECK_CUDA(cudaEventRecord(eventA, stream1));

        // stream2 waits for eventA
        CHECK_CUDA(cudaStreamWaitEvent(stream2, eventA, 0));

        // Launch kernelB on stream1
        float *d_data_half = d_data + halfN;
        kernelB<<<halfBlocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, halfN);

        // Launch kernelC on stream2
        kernelC<<<halfBlocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data_half, halfN);

        // Synchronize both streams
        CHECK_CUDA(cudaEventRecord(eventB, stream2));
        CHECK_CUDA(cudaStreamWaitEvent(stream1, eventB, 0));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float traditionalTime = 0;
    CHECK_CUDA(cudaEventElapsedTime(&traditionalTime, startEvent, stopEvent));

    printf("✓ Traditional launches complete\n");
    printf("  Total time: %.3f ms\n", traditionalTime);
    printf("  Average time per iteration: %.3f μs\n", traditionalTime * 1000.0f / NUM_ITERATIONS);
    printf("  Total API calls: %d (3 launches + 2 events + 2 waits per iteration)\n\n", NUM_ITERATIONS * 7);

    // Copy results and verify
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    bool traditionalCorrect = true;
    for (int i = 0; i < N; i++) {
        float expected;
        if (i < halfN) {
            expected = (float)i * 2.0f + 10.0f;
        } else {
            expected = (float)i * 2.0f + 20.0f;
        }
        if (h_data[i] != expected) {
            traditionalCorrect = false;
            break;
        }
    }
    printf("Traditional result: %s\n\n", traditionalCorrect ? "PASSED" : "FAILED");

    // ========================================
    // Benchmark 2: CUDA Graph with PDL
    // ========================================
    printf("=== Benchmark 2: CUDA Graph with PDL (%d iterations) ===\n", NUM_ITERATIONS);

    // Reset data
    CHECK_CUDA(cudaMemcpy(d_data, h_data_backup, size, cudaMemcpyHostToDevice));

    printf("Capturing graph from stream execution...\n");

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Launch kernelA on stream1
    kernelA<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);

    // Record event after kernelA for cross-stream synchronization
    CHECK_CUDA(cudaEventRecord(eventA, stream1));

    // stream2 waits for eventA
    CHECK_CUDA(cudaStreamWaitEvent(stream2, eventA, 0));

    // Launch kernelB on stream1
    kernelB<<<halfBlocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, halfN);

    // Launch kernelC on stream2
    float *d_data_half = d_data + halfN;
    kernelC<<<halfBlocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data_half, halfN);

    // Synchronize both streams before ending capture
    CHECK_CUDA(cudaEventRecord(eventB, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, eventB, 0));

    // End capture
    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));

    printf("✓ Graph captured successfully!\n");

    printf("Configuring Programmatic Dependent Launch edges...\n");

    // Get all nodes in the graph
    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));

    cudaGraphNode_t *nodes = (cudaGraphNode_t*)malloc(numNodes * sizeof(cudaGraphNode_t));
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes, &numNodes));

    // Find kernelA, kernelB, and kernelC nodes
    cudaGraphNode_t nodeA = NULL, nodeB = NULL, nodeC = NULL;

    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));

        if (nodeType == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CHECK_CUDA(cudaGraphKernelNodeGetParams(nodes[i], &params));

            if (params.func == (void*)kernelA) {
                nodeA = nodes[i];
            } else if (params.func == (void*)kernelB) {
                nodeB = nodes[i];
            } else if (params.func == (void*)kernelC) {
                nodeC = nodes[i];
            }
        }
    }

    if (!nodeA || !nodeB || !nodeC) {
        fprintf(stderr, "Failed to find kernel nodes in graph\n");
        exit(EXIT_FAILURE);
    }

    // Remove existing edges from nodeA to nodeB and nodeC
    cudaGraphNode_t fromNodesRemove[] = {nodeA, nodeA};
    cudaGraphNode_t toNodesRemove[] = {nodeB, nodeC};
    CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromNodesRemove, toNodesRemove, NULL, 2));

    // Add PDL edges
    cudaGraphEdgeData edgeDataB;
    memset(&edgeDataB, 0, sizeof(edgeDataB));
    edgeDataB.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeDataB.to_port = cudaGraphKernelNodePortDefault;
    edgeDataB.type = cudaGraphDependencyTypeProgrammatic;

    cudaGraphEdgeData edgeDataC;
    memset(&edgeDataC, 0, sizeof(edgeDataC));
    edgeDataC.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeDataC.to_port = cudaGraphKernelNodePortDefault;
    edgeDataC.type = cudaGraphDependencyTypeProgrammatic;

    cudaGraphNode_t fromNodesAdd[] = {nodeA, nodeA};
    cudaGraphNode_t toNodesAdd[] = {nodeB, nodeC};
    cudaGraphEdgeData edgeDataArray[] = {edgeDataB, edgeDataC};

    CHECK_CUDA(cudaGraphAddDependencies(graph, fromNodesAdd, toNodesAdd, edgeDataArray, 2));

    printf("✓ PDL edges configured successfully!\n\n");

    free(nodes);

    printf("Instantiating graph...\n");
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("✓ Graph instantiated successfully!\n\n");

    printf("Launching graph %d times...\n", NUM_ITERATIONS);

    CHECK_CUDA(cudaEventRecord(startEvent));

    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
    }

    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float graphTime = 0;
    CHECK_CUDA(cudaEventElapsedTime(&graphTime, startEvent, stopEvent));

    printf("✓ Graph launches complete\n");
    printf("  Total time: %.3f ms\n", graphTime);
    printf("  Average time per iteration: %.3f μs\n", graphTime * 1000.0f / NUM_ITERATIONS);
    printf("  Total API calls: %d (1 launch per iteration)\n\n", NUM_ITERATIONS);

    // Copy results and verify
    CHECK_CUDA(cudaStreamSynchronize(stream1));
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    bool graphCorrect = true;
    for (int i = 0; i < N; i++) {
        float expected;
        if (i < halfN) {
            expected = (float)i * 2.0f + 10.0f;
        } else {
            expected = (float)i * 2.0f + 20.0f;
        }
        if (h_data[i] != expected) {
            graphCorrect = false;
            break;
        }
    }
    printf("Graph result: %s\n\n", graphCorrect ? "PASSED" : "FAILED");

    // ========================================
    // Performance Comparison
    // ========================================
    printf("=== Performance Comparison ===\n");
    printf("Traditional approach: %.3f ms (%.3f μs/iter)\n", traditionalTime, traditionalTime * 1000.0f / NUM_ITERATIONS);
    printf("CUDA Graph approach:  %.3f ms (%.3f μs/iter)\n", graphTime, graphTime * 1000.0f / NUM_ITERATIONS);
    printf("Speedup: %.2fx\n", traditionalTime / graphTime);
    printf("Time saved: %.3f ms\n\n", traditionalTime - graphTime);

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaEventDestroy(eventA));
    CHECK_CUDA(cudaEventDestroy(eventB));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);
    free(h_data_backup);

    printf("=== Summary ===\n");
    printf("This benchmark demonstrates the performance benefit of CUDA Graphs with PDL:\n");
    printf("- Traditional: 7 CUDA API calls per iteration (3 launches + 2 events + 2 waits)\n");
    printf("- CUDA Graph: 1 CUDA API call per iteration (single graph launch)\n");
    printf("- PDL enables kernelB and kernelC to launch early before kernelA completes\n");
    printf("- Multi-stream parallelism is preserved within the graph\n");

    return 0;
}
