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

// LUT size for independent preamble work
#define LUT_SIZE 8192  // 32KB per block

// Producer kernel: Generates output data
__global__ void producerKernel(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Heavy computation to produce results
        float result = input[idx];

        for (int iter = 0; iter < 1000; iter++) {
            result = result * 1.01f + 0.001f;
            result = sqrtf(result * result + 1.0f);
            result = result * 0.99f;
        }

        output[idx] = result;
    }

    // PDL trigger at 50% completion
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x / 2) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Consumer kernel 1: Processes producer output (e.g., activation function)
__global__ void consumer1Kernel(
    const float *producer_output,
    const float *weights1,
    float *consumer1_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table 1
    // ========================================================================
    __shared__ float lut1[LUT_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load LUT1 from global memory (independent of producer)
    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut1[i] = weights1[i];
        lut1[i] = lut1[i] * 2.0f + 1.0f;
    }

    __syncthreads();

    // More independent setup
    float independent_result1 = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result1 += lut1[tid % LUT_SIZE] * (i + 1);
    }

    // ========================================================================
    // WAIT FOR PRODUCER
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Use producer's output
    // ========================================================================
    if (idx < n) {
        float value = producer_output[idx];
        int lut_idx = (int)(value * 100.0f) % LUT_SIZE;
        float transformed = value * lut1[lut_idx] + independent_result1 * 0.0001f;
        consumer1_output[idx] = transformed;
    }
}

// Consumer kernel 2: Processes producer output (e.g., gradient computation)
__global__ void consumer2Kernel(
    const float *producer_output,
    const float *weights2,
    float *consumer2_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table 2
    // ========================================================================
    __shared__ float lut2[LUT_SIZE];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Load LUT2 from global memory (independent of producer)
    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut2[i] = weights2[i];
        lut2[i] = lut2[i] * 1.5f + 0.5f;  // Different transformation
    }

    __syncthreads();

    // More independent setup
    float independent_result2 = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result2 += lut2[tid % LUT_SIZE] * (i + 2);
    }

    // ========================================================================
    // WAIT FOR PRODUCER
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Use producer's output
    // ========================================================================
    if (idx < n) {
        float value = producer_output[idx];
        int lut_idx = (int)(value * 150.0f) % LUT_SIZE;
        float transformed = value * lut2[lut_idx] + independent_result2 * 0.0002f;
        consumer2_output[idx] = transformed;
    }
}

// Baseline kernels without PDL
__global__ void consumer1KernelNoPDL(
    const float *producer_output,
    const float *weights1,
    float *consumer1_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float lut1[LUT_SIZE];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut1[i] = weights1[i];
        lut1[i] = lut1[i] * 2.0f + 1.0f;
    }

    __syncthreads();

    float independent_result1 = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result1 += lut1[tid % LUT_SIZE] * (i + 1);
    }

    if (idx < n) {
        float value = producer_output[idx];
        int lut_idx = (int)(value * 100.0f) % LUT_SIZE;
        float transformed = value * lut1[lut_idx] + independent_result1 * 0.0001f;
        consumer1_output[idx] = transformed;
    }
}

__global__ void consumer2KernelNoPDL(
    const float *producer_output,
    const float *weights2,
    float *consumer2_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float lut2[LUT_SIZE];
    int tid = threadIdx.x;
    int stride = blockDim.x;

    for (int i = tid; i < LUT_SIZE; i += stride) {
        lut2[i] = weights2[i];
        lut2[i] = lut2[i] * 1.5f + 0.5f;
    }

    __syncthreads();

    float independent_result2 = 0.0f;
    for (int i = 0; i < 100; i++) {
        independent_result2 += lut2[tid % LUT_SIZE] * (i + 2);
    }

    if (idx < n) {
        float value = producer_output[idx];
        int lut_idx = (int)(value * 150.0f) % LUT_SIZE;
        float transformed = value * lut2[lut_idx] + independent_result2 * 0.0002f;
        consumer2_output[idx] = transformed;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int SM_COUNT = prop.multiProcessorCount;

    printf("=== PDL with TWO CONSUMER KERNELS ===\n");
    printf("GPU: %s with %d SMs\n\n", prop.name, SM_COUNT);

    printf("Pattern: 1 Producer → 2 Consumers (both with independent LUT loading)\n");
    printf("  - Producer: Heavy computation (~43 μs)\n");
    printf("  - Consumer 1: Loads LUT1 (32KB) + transforms output\n");
    printf("  - Consumer 2: Loads LUT2 (32KB) + transforms output\n");
    printf("  - PDL Benefit: Both consumers start loading LUTs during producer tail\n\n");

    // Configure for 1.5 waves
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)(SM_COUNT * 1.5);
    int N = blocksPerGrid * threadsPerBlock;
    int size = N * sizeof(float);

    printf("Configuration: %d blocks (%.2f waves), %d elements\n\n",
           blocksPerGrid, (float)blocksPerGrid / SM_COUNT, N);

    // Allocate memory
    float *h_input = (float*)malloc(size);
    float *h_weights1 = (float*)malloc(LUT_SIZE * sizeof(float));
    float *h_weights2 = (float*)malloc(LUT_SIZE * sizeof(float));

    for (int i = 0; i < N; i++) h_input[i] = 1.0f + (i % 100) * 0.01f;
    for (int i = 0; i < LUT_SIZE; i++) {
        h_weights1[i] = 2.0f + (i % 1000) * 0.001f;
        h_weights2[i] = 3.0f + (i % 1500) * 0.0015f;
    }

    float *d_input, *d_producer_out, *d_consumer1_out, *d_consumer2_out;
    float *d_weights1, *d_weights2;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_producer_out, size));
    CHECK_CUDA(cudaMalloc(&d_consumer1_out, size));
    CHECK_CUDA(cudaMalloc(&d_consumer2_out, size));
    CHECK_CUDA(cudaMalloc(&d_weights1, LUT_SIZE * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_weights2, LUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights1, h_weights1, LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights2, h_weights2, LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create 3 streams
    cudaStream_t stream1, stream2, stream3;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));

    cudaEvent_t event1, event2, event3;
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventCreate(&event2));
    CHECK_CUDA(cudaEventCreate(&event3));

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    const int NUM_WARMUP = 5;
    const int NUM_ITERS = 20;

    // ========================================================================
    // Baseline: WITHOUT PDL
    // ========================================================================
    printf("=== Baseline: WITHOUT PDL ===\n");

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        producerKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input, d_producer_out, N);
        consumer1KernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_producer_out, d_weights1, d_consumer1_out, N);
        consumer2KernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_producer_out, d_weights2, d_consumer2_out, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CHECK_CUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < NUM_ITERS; i++) {
        producerKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input, d_producer_out, N);
        consumer1KernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_producer_out, d_weights1, d_consumer1_out, N);
        consumer2KernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_producer_out, d_weights2, d_consumer2_out, N);
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float timeNoPDL = 0;
    CHECK_CUDA(cudaEventElapsedTime(&timeNoPDL, startEvent, stopEvent));

    printf("  Total time: %.3f ms\n", timeNoPDL);
    printf("  Average per iteration: %.3f ms\n", timeNoPDL / NUM_ITERS);
    printf("  Pattern: Producer → Consumer1 → Consumer2 (sequential)\n\n");

    // ========================================================================
    // WITH PDL: 1 Producer → 2 Consumers
    // ========================================================================
    printf("=== WITH PDL: 1 Producer → 2 Consumers ===\n");

    printf("Capturing graph with PDL edges...\n");

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Producer on stream1
    producerKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input, d_producer_out, N);

    // Fork to two consumer streams
    CHECK_CUDA(cudaEventRecord(event1, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, event1, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event1, 0));

    // Consumer 1 on stream2
    consumer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(
        d_producer_out, d_weights1, d_consumer1_out, N);

    // Consumer 2 on stream3
    consumer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(
        d_producer_out, d_weights2, d_consumer2_out, N);

    // Join back to stream1
    CHECK_CUDA(cudaEventRecord(event2, stream2));
    CHECK_CUDA(cudaEventRecord(event3, stream3));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event2, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event3, 0));

    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
    printf("✓ Graph captured\n");

    // Configure PDL edges
    printf("Configuring PDL edges...\n");

    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    cudaGraphNode_t *nodes = (cudaGraphNode_t*)malloc(numNodes * sizeof(cudaGraphNode_t));
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes, &numNodes));

    cudaGraphNode_t producerNode = NULL, consumer1Node = NULL, consumer2Node = NULL;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));
        if (nodeType == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CHECK_CUDA(cudaGraphKernelNodeGetParams(nodes[i], &params));
            if (params.func == (void*)producerKernel) producerNode = nodes[i];
            else if (params.func == (void*)consumer1Kernel) consumer1Node = nodes[i];
            else if (params.func == (void*)consumer2Kernel) consumer2Node = nodes[i];
        }
    }

    if (!producerNode || !consumer1Node || !consumer2Node) {
        fprintf(stderr, "Failed to find kernel nodes\n");
        exit(EXIT_FAILURE);
    }

    // Remove default edges
    cudaGraphNode_t fromRemove[] = {producerNode, producerNode};
    cudaGraphNode_t toRemove[] = {consumer1Node, consumer2Node};
    CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromRemove, toRemove, NULL, 2));

    // Add PDL edges: Producer → Consumer1 and Producer → Consumer2
    cudaGraphEdgeData edgeData1, edgeData2;
    memset(&edgeData1, 0, sizeof(edgeData1));
    memset(&edgeData2, 0, sizeof(edgeData2));

    edgeData1.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData1.to_port = cudaGraphKernelNodePortDefault;
    edgeData1.type = cudaGraphDependencyTypeProgrammatic;

    edgeData2.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData2.to_port = cudaGraphKernelNodePortDefault;
    edgeData2.type = cudaGraphDependencyTypeProgrammatic;

    cudaGraphNode_t fromAdd[] = {producerNode, producerNode};
    cudaGraphNode_t toAdd[] = {consumer1Node, consumer2Node};
    cudaGraphEdgeData edgeArray[] = {edgeData1, edgeData2};

    CHECK_CUDA(cudaGraphAddDependencies(graph, fromAdd, toAdd, edgeArray, 2));
    printf("✓ PDL edges configured: Producer → Consumer1, Producer → Consumer2\n\n");

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
    printf("  Average per iteration: %.3f ms\n", timeWithPDL / NUM_ITERS);
    printf("  Pattern: Producer triggers → Consumer1 & Consumer2 start concurrently\n\n");

    // ========================================================================
    // Results comparison
    // ========================================================================
    printf("=== Performance Comparison ===\n");
    printf("  WITHOUT PDL: %.3f ms (%.3f ms/iter)\n", timeNoPDL, timeNoPDL / NUM_ITERS);
    printf("  WITH PDL:    %.3f ms (%.3f ms/iter)\n", timeWithPDL, timeWithPDL / NUM_ITERS);

    if (timeWithPDL < timeNoPDL) {
        float speedup = timeNoPDL / timeWithPDL;
        float saved = timeNoPDL - timeWithPDL;
        printf("\n  ✓ PDL BENEFIT: %.2fx speedup (%.3f ms saved)\n", speedup, saved);
        printf("  Both consumers launch when producer triggers (50%% done)\n");
        printf("  Consumer1 & Consumer2 can overlap with producer tail!\n");
    } else {
        printf("\n  Note: PDL benefit may be small on this workload\n");
    }

    printf("\n=== Profile with Nsight Systems ===\n");
    printf("nsys profile -t cuda --cuda-graph-trace=node \\\n");
    printf("             -o pdl_two_consumers ./cuda_pdl_two_consumers\n\n");
    printf("Expected timeline:\n");
    printf("  Producer (43μs) ────────────────────|\n");
    printf("            50%% ↓                     |\n");
    printf("            Consumer1 loads LUT1 ─────┤ overlap\n");
    printf("            Consumer2 loads LUT2 ─────┤ overlap\n");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaEventDestroy(event2));
    CHECK_CUDA(cudaEventDestroy(event3));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_producer_out));
    CHECK_CUDA(cudaFree(d_consumer1_out));
    CHECK_CUDA(cudaFree(d_consumer2_out));
    CHECK_CUDA(cudaFree(d_weights1));
    CHECK_CUDA(cudaFree(d_weights2));
    free(h_input);
    free(h_weights1);
    free(h_weights2);

    return 0;
}
