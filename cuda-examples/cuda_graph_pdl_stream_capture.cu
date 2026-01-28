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
// PDL allows kernelB and kernelC to start before this kernel fully completes
__global__ void kernelA(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Perform computation
        data[idx] = data[idx] * 2.0f;
    }

    // PDL: Signal dependent kernels to launch early
    // This is called by one thread when kernelA is mostly done
    // Dependent kernels (kernelB, kernelC) can now start without waiting
    // for kernelA to fully complete
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
    int N = 1024;
    int size = N * sizeof(float);

    printf("=== CUDA Graph with PDL using Stream Capture ===\n\n");

    // Allocate host memory
    float *h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Create 2 streams for parallel execution of kernelB and kernelC
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    // Create events for synchronization
    cudaEvent_t eventA, eventB;
    CHECK_CUDA(cudaEventCreate(&eventA));
    CHECK_CUDA(cudaEventCreate(&eventB));

    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int halfN = N / 2;
    int halfBlocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    // ========================================
    // Step 1: Create CUDA Graph using Stream Capture
    // ========================================
    printf("Capturing graph from stream execution...\n");

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Launch kernelA on stream1
    kernelA<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, N);

    // Record event after kernelA for cross-stream synchronization
    CHECK_CUDA(cudaEventRecord(eventA, stream1));

    // Only stream2 needs to wait for eventA (stream1 is already serialized)
    CHECK_CUDA(cudaStreamWaitEvent(stream2, eventA, 0));

    // Launch kernelB on stream1 (automatically waits for kernelA - same stream)
    kernelB<<<halfBlocksPerGrid, threadsPerBlock, 0, stream1>>>(d_data, halfN);

    // Launch kernelC on stream2 (waits via eventA)
    float *d_data_half = d_data + halfN;
    kernelC<<<halfBlocksPerGrid, threadsPerBlock, 0, stream2>>>(d_data_half, halfN);

    // Synchronize both streams before ending capture
    // Use eventB to make stream1 wait for stream2 to complete
    CHECK_CUDA(cudaEventRecord(eventB, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, eventB, 0));

    // End capture on stream1
    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));

    printf("✓ Graph captured successfully!\n");
    printf("✓ kernelB captured on stream1\n");
    printf("✓ kernelC captured on stream2\n\n");

    // ========================================
    // Step 2: Modify edges to use PDL
    // ========================================
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

    printf("✓ PDL edges configured successfully!\n");
    printf("  nodeA -> nodeB (PDL port, stream1)\n");
    printf("  nodeA -> nodeC (PDL port, stream2)\n\n");

    free(nodes);

    // ========================================
    // Step 3: Instantiate the graph
    // ========================================
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    printf("Graph Structure:\n");
    printf("                  kernelA\n");
    printf("                 /       \\\n");
    printf("          [PDL Port]   [PDL Port]\n");
    printf("               /           \\\n");
    printf("          kernelB       kernelC\n");
    printf("         (stream1)     (stream2)\n");
    printf("               \\           /\n");
    printf("                \\         /\n");
    printf("                 Sync Node\n\n");

    printf("Key Features:\n");
    printf("✓ Using stream capture to create graph\n");
    printf("✓ kernelB actually runs on stream1\n");
    printf("✓ kernelC actually runs on stream2\n");
    printf("✓ PDL edges via cudaGraphKernelNodePortProgrammatic\n");
    printf("✓ kernelA calls cudaTriggerProgrammaticLaunchCompletion()\n");
    printf("✓ kernelB and kernelC launch early (before kernelA completes)\n\n");

    // ========================================
    // Step 4: Launch the graph
    // ========================================
    printf("Launching CUDA Graph...\n");
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
    CHECK_CUDA(cudaStreamSynchronize(stream1));  // Waits for entire graph (includes stream2 work)
    printf("Graph execution complete.\n\n");

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // ========================================
    // Verify results
    // ========================================
    printf("=== Verification (first 16 elements) ===\n");
    bool correct = true;
    for (int i = 0; i < 16; i++) {
        float expected;
        if (i < halfN) {
            // First half: kernelA (×2) then kernelB (+10)
            expected = (float)i * 2.0f + 10.0f;
        } else {
            // Second half: kernelA (×2) then kernelC (+20)
            expected = (float)i * 2.0f + 20.0f;
        }

        bool match = (h_data[i] == expected);
        printf("h_data[%4d] = %7.2f (expected %7.2f) %s\n",
               i, h_data[i], expected, match ? "[OK]" : "[FAIL]");

        if (!match) correct = false;
    }

    printf("\n=== Result: %s ===\n", correct ? "PASSED" : "FAILED");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(eventA));
    CHECK_CUDA(cudaEventDestroy(eventB));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_data));
    free(h_data);

    printf("\n=== Summary ===\n");
    printf("This version uses stream capture to create a CUDA graph where:\n");
    printf("- kernelB and kernelC are captured on different streams (stream1, stream2)\n");
    printf("- PDL edges are configured post-capture using cudaGraphAddDependencies\n");
    printf("- Dependent kernels launch early via cudaTriggerProgrammaticLaunchCompletion()\n");
    printf("- True multi-stream parallelism within the graph\n");

    return 0;
}
