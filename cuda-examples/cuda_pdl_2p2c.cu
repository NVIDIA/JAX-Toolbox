#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define LUT_SIZE 8192  // 32KB

// ============================================================================
// SHARED BUFFER APPROACH:
// Both producers write to DIFFERENT PARTS of the SAME memory buffer
// Producer1: writes to shared_output[0 .. n/2-1]      (first half)
// Producer2: writes to shared_output[n/2 .. n-1]      (second half)
// Consumers: poll/read from BOTH parts of the shared buffer
// ============================================================================

// Producer 1: Writes to FIRST HALF of shared buffer [0 .. n/2-1]
__global__ void producer1Kernel(
    const float *input,
    float *shared_output,  // SHARED buffer
    int n,
    int offset             // Starting offset for this producer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = offset + idx;  // Offset into shared buffer

    if (idx < n) {
        float result = input[idx];

        // Heavy computation for producer 1
        for (int iter = 0; iter < 800; iter++) {
            result = result * 1.01f + 0.001f;
            result = sqrtf(result * result + 1.0f);
            result = result * 0.99f;
        }

        // Write to FIRST HALF of shared buffer
        shared_output[global_idx] = result;
    }

    // PDL trigger at 50%
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x / 2) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Producer 2: Writes to SECOND HALF of shared buffer [n/2 .. n-1]
__global__ void producer2Kernel(
    const float *input,
    float *shared_output,  // SHARED buffer (same as Producer1)
    int n,
    int offset             // Starting offset for this producer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_idx = offset + idx;  // Offset into shared buffer

    if (idx < n) {
        float result = input[idx];

        // Heavy computation for producer 2 (slightly different)
        for (int iter = 0; iter < 800; iter++) {
            result = result * 1.02f + 0.002f;
            result = sqrtf(result * result + 0.5f);
            result = result * 0.98f;
        }

        // Write to SECOND HALF of shared buffer
        shared_output[global_idx] = result;
    }

    // PDL trigger at 50%
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x / 2) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Consumer 1: Polls/reads from BOTH PARTS of the shared buffer
__global__ void consumer1Kernel(
    const float *shared_output,  // Read from SHARED buffer (both halves)
    const float *weights,
    float *final_output1,
    int total_n                  // Total size of shared buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table
    // This can happen while producers are still running
    // ========================================================================
    __shared__ float lut[LUT_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut[i] = weights[i];
        lut[i] = lut[i] * 2.0f + 1.0f;
    }

    __syncthreads();

    float independent_result = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result += lut[tid % LUT_SIZE] * (i + 1);
    }

    // ========================================================================
    // WAIT FOR BOTH PRODUCERS!
    // Both must finish writing their parts of the shared buffer
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Poll/read from BOTH PARTS of shared buffer
    // ========================================================================
    if (idx < total_n) {
        int half_n = total_n / 2;

        // Read from FIRST HALF (Producer1's part)
        int idx1 = idx % half_n;
        float value1 = shared_output[idx1];

        // Read from SECOND HALF (Producer2's part)
        int idx2 = half_n + (idx % half_n);
        float value2 = shared_output[idx2];

        // Combine data from both producers
        float combined = value1 + value2;

        int lut_idx = (int)(combined * 50.0f) % LUT_SIZE;
        float transformed = combined * lut[lut_idx] + independent_result * 0.0001f;

        final_output1[idx] = transformed;
    }
}

// Consumer 2: Also polls/reads from BOTH PARTS of the shared buffer
__global__ void consumer2Kernel(
    const float *shared_output,  // Read from SHARED buffer (both halves)
    const float *weights,
    float *final_output2,
    int total_n                  // Total size of shared buffer
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table (different transformation)
    // This can happen while producers are still running
    // ========================================================================
    __shared__ float lut[LUT_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut[i] = weights[i];
        lut[i] = lut[i] * 3.0f + 2.0f;  // Different transformation
    }

    __syncthreads();

    float independent_result = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result += lut[tid % LUT_SIZE] * (i + 2);
    }

    // ========================================================================
    // WAIT FOR BOTH PRODUCERS!
    // Both must finish writing their parts of the shared buffer
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Poll/read from BOTH PARTS of shared buffer
    // ========================================================================
    if (idx < total_n) {
        int half_n = total_n / 2;

        // Read from FIRST HALF (Producer1's part)
        int idx1 = idx % half_n;
        float value1 = shared_output[idx1];

        // Read from SECOND HALF (Producer2's part)
        int idx2 = half_n + (idx % half_n);
        float value2 = shared_output[idx2];

        // Combine data from both producers (different formula)
        float combined = value1 * 1.5f + value2 * 0.5f;

        int lut_idx = (int)(combined * 60.0f) % LUT_SIZE;
        float transformed = combined * lut[lut_idx] + independent_result * 0.0002f;

        final_output2[idx] = transformed;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int SM_COUNT = prop.multiProcessorCount;

    printf("=== PDL 2P2C: SHARED BUFFER APPROACH ===\n");
    printf("GPU: %s with %d SMs\n\n", prop.name, SM_COUNT);

    printf("Memory Pattern: SHARED BUFFER with PARTITIONING\n");
    printf("  Producer1 → writes to shared_output[0 .. N/2-1]      (first half)\n");
    printf("  Producer2 → writes to shared_output[N/2 .. N-1]      (second half)\n");
    printf("  Consumer1 → polls/reads from BOTH halves\n");
    printf("  Consumer2 → polls/reads from BOTH halves\n\n");

    printf("Dependency Pattern:\n");
    printf("  Producer1 ─┬─→ Consumer1 (PDL edge)\n");
    printf("             │\n");
    printf("  Producer2 ─┼─→ Consumer1 (PDL edge)\n");
    printf("             │\n");
    printf("  Producer1 ─┤\n");
    printf("             │\n");
    printf("  Producer2 ─┴─→ Consumer2 (PDL edge)\n\n");
    printf("  Each consumer depends on BOTH producers\n\n");

    // Configure for 1.5 waves
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)(SM_COUNT * 1.5);
    int half_N = blocksPerGrid * threadsPerBlock / 2;  // Each producer processes half
    int total_N = blocksPerGrid * threadsPerBlock;     // Total buffer size
    int half_size = half_N * sizeof(float);
    int total_size = total_N * sizeof(float);

    printf("Configuration:\n");
    printf("  Total blocks: %d (%.2f waves)\n", blocksPerGrid, (float)blocksPerGrid / SM_COUNT);
    printf("  Total elements: %d\n", total_N);
    printf("  Producer1 range: [0 .. %d)  (%d elements)\n", half_N, half_N);
    printf("  Producer2 range: [%d .. %d) (%d elements)\n\n", half_N, total_N, half_N);

    // Allocate memory
    float *h_input1 = (float*)malloc(half_size);
    float *h_input2 = (float*)malloc(half_size);
    float *h_weights = (float*)malloc(LUT_SIZE * sizeof(float));

    for (int i = 0; i < half_N; i++) {
        h_input1[i] = 1.0f + (i % 100) * 0.01f;
        h_input2[i] = 2.0f + (i % 150) * 0.015f;
    }
    for (int i = 0; i < LUT_SIZE; i++) {
        h_weights[i] = 2.0f + (i % 1000) * 0.001f;
    }

    // Device memory: SHARED OUTPUT BUFFER
    float *d_input1, *d_input2, *d_shared_output;  // ONE shared buffer!
    float *d_final_out1, *d_final_out2, *d_weights;
    CHECK_CUDA(cudaMalloc(&d_input1, half_size));
    CHECK_CUDA(cudaMalloc(&d_input2, half_size));
    CHECK_CUDA(cudaMalloc(&d_shared_output, total_size));  // SHARED buffer (full size)
    CHECK_CUDA(cudaMalloc(&d_final_out1, total_size));
    CHECK_CUDA(cudaMalloc(&d_final_out2, total_size));
    CHECK_CUDA(cudaMalloc(&d_weights, LUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input1, h_input1, half_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input2, h_input2, half_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create 4 streams for producers and consumers
    cudaStream_t stream1, stream2, stream3, stream4;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));
    CHECK_CUDA(cudaStreamCreate(&stream4));

    cudaEvent_t event1, event2, event3, event4;
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventCreate(&event2));
    CHECK_CUDA(cudaEventCreate(&event3));
    CHECK_CUDA(cudaEventCreate(&event4));

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    const int NUM_WARMUP = 5;
    const int NUM_ITERS = 20;

    // ========================================================================
    // WITH PDL: Shared Buffer Approach
    // ========================================================================
    printf("=== WITH PDL: Shared Buffer with Partitioning ===\n");
    printf("Capturing graph with PDL edges...\n");

    cudaGraph_t graph;
    cudaEvent_t forkEvent;
    CHECK_CUDA(cudaEventCreate(&forkEvent));

    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Fork to stream2 for parallel producers
    CHECK_CUDA(cudaEventRecord(forkEvent, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, forkEvent, 0));

    // Producer1 writes to FIRST HALF [0 .. half_N-1], Producer2 writes to SECOND HALF [half_N .. total_N-1]
    int blocks_per_producer = blocksPerGrid / 2;
    producer1Kernel<<<blocks_per_producer, threadsPerBlock, 0, stream1>>>(
        d_input1, d_shared_output, half_N, 0);           // offset = 0 (first half)
    producer2Kernel<<<blocks_per_producer, threadsPerBlock, 0, stream2>>>(
        d_input2, d_shared_output, half_N, half_N);     // offset = half_N (second half)

    // Consumer1 depends on BOTH producers
    CHECK_CUDA(cudaEventRecord(event1, stream1));
    CHECK_CUDA(cudaEventRecord(event2, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event1, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event2, 0));
    consumer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(
        d_shared_output, d_weights, d_final_out1, total_N);  // Read from SHARED buffer

    // Consumer2 also depends on BOTH producers
    CHECK_CUDA(cudaStreamWaitEvent(stream4, event1, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream4, event2, 0));
    consumer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream4>>>(
        d_shared_output, d_weights, d_final_out2, total_N);  // Read from SHARED buffer

    // Sync all back to stream1
    CHECK_CUDA(cudaEventRecord(event3, stream3));
    CHECK_CUDA(cudaEventRecord(event4, stream4));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event3, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event4, 0));

    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
    printf("✓ Graph captured\n");

    // Configure PDL edges
    printf("Configuring PDL edges for shared buffer...\n");

    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    cudaGraphNode_t *nodes = (cudaGraphNode_t*)malloc(numNodes * sizeof(cudaGraphNode_t));
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes, &numNodes));

    cudaGraphNode_t producer1Node = NULL, producer2Node = NULL;
    cudaGraphNode_t consumer1Node = NULL, consumer2Node = NULL;

    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));
        if (nodeType == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CHECK_CUDA(cudaGraphKernelNodeGetParams(nodes[i], &params));
            if (params.func == (void*)producer1Kernel) producer1Node = nodes[i];
            else if (params.func == (void*)producer2Kernel) producer2Node = nodes[i];
            else if (params.func == (void*)consumer1Kernel) consumer1Node = nodes[i];
            else if (params.func == (void*)consumer2Kernel) consumer2Node = nodes[i];
        }
    }

    if (!producer1Node || !producer2Node || !consumer1Node || !consumer2Node) {
        fprintf(stderr, "Failed to find kernel nodes\n");
        exit(EXIT_FAILURE);
    }

    // Remove default edges (4 edges total)
    cudaGraphNode_t fromRemove[] = {producer1Node, producer2Node, producer1Node, producer2Node};
    cudaGraphNode_t toRemove[] = {consumer1Node, consumer1Node, consumer2Node, consumer2Node};
    CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromRemove, toRemove, NULL, 4));

    // Add FOUR PDL edges (each consumer depends on both producers)
    cudaGraphEdgeData edgeData1, edgeData2, edgeData3, edgeData4;
    memset(&edgeData1, 0, sizeof(edgeData1));
    memset(&edgeData2, 0, sizeof(edgeData2));
    memset(&edgeData3, 0, sizeof(edgeData3));
    memset(&edgeData4, 0, sizeof(edgeData4));

    edgeData1.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData1.to_port = cudaGraphKernelNodePortDefault;
    edgeData1.type = cudaGraphDependencyTypeProgrammatic;

    edgeData2.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData2.to_port = cudaGraphKernelNodePortDefault;
    edgeData2.type = cudaGraphDependencyTypeProgrammatic;

    edgeData3.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData3.to_port = cudaGraphKernelNodePortDefault;
    edgeData3.type = cudaGraphDependencyTypeProgrammatic;

    edgeData4.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData4.to_port = cudaGraphKernelNodePortDefault;
    edgeData4.type = cudaGraphDependencyTypeProgrammatic;

    cudaGraphNode_t fromAdd[] = {producer1Node, producer2Node, producer1Node, producer2Node};
    cudaGraphNode_t toAdd[] = {consumer1Node, consumer1Node, consumer2Node, consumer2Node};
    cudaGraphEdgeData edgeArray[] = {edgeData1, edgeData2, edgeData3, edgeData4};

    CHECK_CUDA(cudaGraphAddDependencies(graph, fromAdd, toAdd, edgeArray, 4));
    printf("✓ PDL edges configured:\n");
    printf("  Producer1 (writes [0..%d))     → Consumer1 (PDL)\n", half_N);
    printf("  Producer2 (writes [%d..%d)) → Consumer1 (PDL)\n", half_N, total_N);
    printf("  Producer1 (writes [0..%d))     → Consumer2 (PDL)\n", half_N);
    printf("  Producer2 (writes [%d..%d)) → Consumer2 (PDL)\n", half_N, total_N);
    printf("  Each consumer polls both parts of shared buffer!\n\n");

    free(nodes);

    // Instantiate graph
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CHECK_CUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < NUM_ITERS; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream1));
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float timeWithPDL = 0;
    CHECK_CUDA(cudaEventElapsedTime(&timeWithPDL, startEvent, stopEvent));

    printf("  Total time: %.3f ms\n", timeWithPDL);
    printf("  Average per iteration: %.3f ms\n\n", timeWithPDL / NUM_ITERS);

    // ========================================================================
    // Results
    // ========================================================================
    printf("=== Shared Buffer Benefits ===\n");
    printf("This approach demonstrates:\n");
    printf("  ✓ Realistic memory pattern (matrix tiles, data streams)\n");
    printf("  ✓ Both producers write to NON-OVERLAPPING parts\n");
    printf("  ✓ Consumers poll from both parts after PDL triggers\n");
    printf("  ✓ More cache-friendly (single contiguous buffer)\n\n");

    printf("=== Expected Behavior ===\n");
    printf("Each consumer has 2 incoming PDL edges:\n");
    printf("  - Must wait for BOTH producers to trigger\n");
    printf("  - Expected: NO overlap (multi-dependency semantics)\n\n");

    printf("=== Profile with Nsight Systems ===\n");
    printf("nsys profile -t cuda --cuda-graph-trace=node \\\n");
    printf("             -o pdl_2p2c ./cuda_pdl_2p2c\n\n");
    printf("Look for:\n");
    printf("  1. Both producers running in parallel\n");
    printf("  2. Both writing to SAME buffer (different regions)\n");
    printf("  3. Consumers starting AFTER both producers trigger\n");
    printf("  4. NO memory conflicts (non-overlapping writes)\n");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(forkEvent));
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaEventDestroy(event2));
    CHECK_CUDA(cudaEventDestroy(event3));
    CHECK_CUDA(cudaEventDestroy(event4));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaStreamDestroy(stream4));
    CHECK_CUDA(cudaFree(d_input1));
    CHECK_CUDA(cudaFree(d_input2));
    CHECK_CUDA(cudaFree(d_shared_output));
    CHECK_CUDA(cudaFree(d_final_out1));
    CHECK_CUDA(cudaFree(d_final_out2));
    CHECK_CUDA(cudaFree(d_weights));
    free(h_input1);
    free(h_input2);
    free(h_weights);

    return 0;
}
