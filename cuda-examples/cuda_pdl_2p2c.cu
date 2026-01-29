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

// Producer 1: Generates first dataset
__global__ void producer1Kernel(const float *input1, float *output1, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float result = input1[idx];

        // Heavy computation for producer 1
        for (int iter = 0; iter < 800; iter++) {
            result = result * 1.01f + 0.001f;
            result = sqrtf(result * result + 1.0f);
            result = result * 0.99f;
        }

        output1[idx] = result;
    }

    // PDL trigger at 50%
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x / 2) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Producer 2: Generates second dataset
__global__ void producer2Kernel(const float *input2, float *output2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        float result = input2[idx];

        // Heavy computation for producer 2
        for (int iter = 0; iter < 800; iter++) {
            result = result * 1.02f + 0.002f;
            result = sqrtf(result * result + 0.5f);
            result = result * 0.98f;
        }

        output2[idx] = result;
    }

    // PDL trigger at 50%
    if (threadIdx.x == 0 && blockIdx.x == gridDim.x / 2) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
}

// Consumer 1: Processes output from Producer 1 ONLY
__global__ void consumer1Kernel(
    const float *producer1_output,
    const float *weights,
    float *final_output1,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table
    // This can happen while Producer1 is still running!
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
    // WAIT FOR PRODUCER 1
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Use output from Producer 1
    // ========================================================================
    if (idx < n) {
        float value1 = producer1_output[idx];

        int lut_idx = (int)(value1 * 50.0f) % LUT_SIZE;
        float transformed = value1 * lut[lut_idx] + independent_result * 0.0001f;

        final_output1[idx] = transformed;
    }
}

// Consumer 2: Processes output from Producer 2 ONLY
__global__ void consumer2Kernel(
    const float *producer2_output,
    const float *weights,
    float *final_output2,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table
    // This can happen while Producer2 is still running!
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
    // WAIT FOR PRODUCER 2
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Use output from Producer 2
    // ========================================================================
    if (idx < n) {
        float value2 = producer2_output[idx];

        int lut_idx = (int)(value2 * 60.0f) % LUT_SIZE;
        float transformed = value2 * lut[lut_idx] + independent_result * 0.0002f;

        final_output2[idx] = transformed;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int SM_COUNT = prop.multiProcessorCount;

    printf("=== PDL with 2 PRODUCERS → 2 CONSUMERS ===\n");
    printf("GPU: %s with %d SMs\n\n", prop.name, SM_COUNT);

    printf("Pattern: Independent 1P1C chains\n");
    printf("  Chain 1: Producer1 → Consumer1\n");
    printf("  Chain 2: Producer2 → Consumer2\n");
    printf("  Expected: BOTH consumers should overlap with their producers\n");
    printf("  Reason: Each consumer has only 1 incoming PDL edge\n\n");

    // Configure for 1.5 waves
    int threadsPerBlock = 256;
    int blocksPerGrid = (int)(SM_COUNT * 1.5);
    int N = blocksPerGrid * threadsPerBlock;
    int size = N * sizeof(float);

    printf("Configuration: %d blocks (%.2f waves), %d elements\n\n",
           blocksPerGrid, (float)blocksPerGrid / SM_COUNT, N);

    // Allocate memory
    float *h_input1 = (float*)malloc(size);
    float *h_input2 = (float*)malloc(size);
    float *h_weights = (float*)malloc(LUT_SIZE * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_input1[i] = 1.0f + (i % 100) * 0.01f;
        h_input2[i] = 2.0f + (i % 150) * 0.015f;
    }
    for (int i = 0; i < LUT_SIZE; i++) {
        h_weights[i] = 2.0f + (i % 1000) * 0.001f;
    }

    float *d_input1, *d_input2, *d_output1, *d_output2;
    float *d_final_out1, *d_final_out2, *d_weights;
    CHECK_CUDA(cudaMalloc(&d_input1, size));
    CHECK_CUDA(cudaMalloc(&d_input2, size));
    CHECK_CUDA(cudaMalloc(&d_output1, size));
    CHECK_CUDA(cudaMalloc(&d_output2, size));
    CHECK_CUDA(cudaMalloc(&d_final_out1, size));
    CHECK_CUDA(cudaMalloc(&d_final_out2, size));
    CHECK_CUDA(cudaMalloc(&d_weights, LUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice));
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
    // WITH PDL: 2 Producers → 2 Consumers (Independent Chains)
    // ========================================================================
    printf("=== WITH PDL: 2P → 2C (Independent Chains) ===\n");

    printf("Capturing graph with PDL edges...\n");

    cudaGraph_t graph;
    cudaEvent_t forkEvent;
    CHECK_CUDA(cudaEventCreate(&forkEvent));

    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Fork to stream2 for parallel producers
    CHECK_CUDA(cudaEventRecord(forkEvent, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, forkEvent, 0));

    // Producer1 on stream1, Producer2 on stream2 (parallel)
    producer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input1, d_output1, N);
    producer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_input2, d_output2, N);

    // Consumer1 depends on Producer1
    CHECK_CUDA(cudaEventRecord(event1, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event1, 0));
    consumer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(
        d_output1, d_weights, d_final_out1, N);

    // Consumer2 depends on Producer2
    CHECK_CUDA(cudaEventRecord(event2, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream4, event2, 0));
    consumer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream4>>>(
        d_output2, d_weights, d_final_out2, N);

    // Sync all back to stream1
    CHECK_CUDA(cudaEventRecord(event3, stream3));
    CHECK_CUDA(cudaEventRecord(event4, stream4));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event3, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event4, 0));

    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
    printf("✓ Graph captured\n");

    // Configure PDL edges
    printf("Configuring PDL edges...\n");

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

    // Remove default edges
    cudaGraphNode_t fromRemove[] = {producer1Node, producer2Node};
    cudaGraphNode_t toRemove[] = {consumer1Node, consumer2Node};
    CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromRemove, toRemove, NULL, 2));

    // Add TWO INDEPENDENT PDL edges
    // Edge 1: Producer1 → Consumer1
    // Edge 2: Producer2 → Consumer2
    cudaGraphEdgeData edgeData1, edgeData2;
    memset(&edgeData1, 0, sizeof(edgeData1));
    memset(&edgeData2, 0, sizeof(edgeData2));

    edgeData1.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData1.to_port = cudaGraphKernelNodePortDefault;
    edgeData1.type = cudaGraphDependencyTypeProgrammatic;

    edgeData2.from_port = cudaGraphKernelNodePortProgrammatic;
    edgeData2.to_port = cudaGraphKernelNodePortDefault;
    edgeData2.type = cudaGraphDependencyTypeProgrammatic;

    cudaGraphNode_t fromAdd[] = {producer1Node, producer2Node};
    cudaGraphNode_t toAdd[] = {consumer1Node, consumer2Node};
    cudaGraphEdgeData edgeArray[] = {edgeData1, edgeData2};

    CHECK_CUDA(cudaGraphAddDependencies(graph, fromAdd, toAdd, edgeArray, 2));
    printf("✓ PDL edges configured:\n");
    printf("  Producer1 → Consumer1 (PDL - independent chain)\n");
    printf("  Producer2 → Consumer2 (PDL - independent chain)\n");
    printf("  Each consumer has only 1 incoming PDL edge!\n\n");

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
    printf("=== Expected Behavior ===\n");
    printf("Each consumer has only 1 incoming PDL edge:\n");
    printf("  - Consumer1 depends only on Producer1\n");
    printf("  - Consumer2 depends only on Producer2\n");
    printf("  - Both should achieve overlap (like 1P1C pattern)\n\n");

    printf("Timeline expectation:\n");
    printf("  Producer1 (38μs) ────────────────┐\n");
    printf("              50%% ↓                 │\n");
    printf("            Consumer1 launches ─────┤ (overlaps)\n");
    printf("  Producer2 (38μs) ────────────────┐│\n");
    printf("              50%% ↓                 ││\n");
    printf("            Consumer2 launches ─────┤│ (overlaps)\n");
    printf("                                    ↓↓\n");
    printf("                  Both producers complete\n");
    printf("                  Consumers finish dependent work\n\n");

    printf("=== Profile with Nsight Systems ===\n");
    printf("nsys profile -t cuda --cuda-graph-trace=node \\\n");
    printf("             -o pdl_2p2c ./cuda_pdl_2p2c\n\n");
    printf("Look for:\n");
    printf("  1. Consumer1 launching at ~50%% of Producer1 execution\n");
    printf("  2. Consumer2 launching at ~50%% of Producer2 execution\n");
    printf("  3. Overlap between consumers and producers\n");
    printf("  4. All 4 kernels potentially running concurrently\n");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
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
    CHECK_CUDA(cudaFree(d_output1));
    CHECK_CUDA(cudaFree(d_output2));
    CHECK_CUDA(cudaFree(d_final_out1));
    CHECK_CUDA(cudaFree(d_final_out2));
    CHECK_CUDA(cudaFree(d_weights));
    free(h_input1);
    free(h_input2);
    free(h_weights);

    return 0;
}
