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

        // Heavy computation for producer 2 (slightly different)
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

// Consumer: Combines outputs from BOTH producers
__global__ void consumerKernel(
    const float *producer1_output,
    const float *producer2_output,
    const float *weights,
    float *final_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // INDEPENDENT WORK: Load lookup table
    // This can happen while producers are still running!
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
    // cudaGridDependencySynchronize() should wait for ALL incoming PDL edges
    // ========================================================================
    cudaGridDependencySynchronize();

    // ========================================================================
    // DEPENDENT WORK: Use outputs from BOTH producers
    // ========================================================================
    if (idx < n) {
        float value1 = producer1_output[idx];  // From producer 1
        float value2 = producer2_output[idx];  // From producer 2

        // Combine both producer outputs
        float combined = value1 + value2;

        int lut_idx = (int)(combined * 50.0f) % LUT_SIZE;
        float transformed = combined * lut[lut_idx] + independent_result * 0.0001f;

        final_output[idx] = transformed;
    }
}

// Baseline kernels without PDL
__global__ void consumerKernelNoPDL(
    const float *producer1_output,
    const float *producer2_output,
    const float *weights,
    float *final_output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

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

    if (idx < n) {
        float value1 = producer1_output[idx];
        float value2 = producer2_output[idx];
        float combined = value1 + value2;
        int lut_idx = (int)(combined * 50.0f) % LUT_SIZE;
        float transformed = combined * lut[lut_idx] + independent_result * 0.0001f;
        final_output[idx] = transformed;
    }
}

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    int SM_COUNT = prop.multiProcessorCount;

    printf("=== PDL with TWO PRODUCERS → ONE CONSUMER ===\n");
    printf("GPU: %s with %d SMs\n\n", prop.name, SM_COUNT);

    printf("Pattern: 2 Producers → 1 Consumer (join pattern)\n");
    printf("  - Producer 1: Heavy computation (~35 μs) → output1\n");
    printf("  - Producer 2: Heavy computation (~35 μs) → output2\n");
    printf("  - Consumer: Loads LUT (32KB) + combines output1 & output2\n");
    printf("  - PDL Benefit: Consumer starts loading LUT while producers finish\n");
    printf("  - Key: cudaGridDependencySynchronize() waits for BOTH producers!\n\n");

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

    float *d_input1, *d_input2, *d_output1, *d_output2, *d_final_out, *d_weights;
    CHECK_CUDA(cudaMalloc(&d_input1, size));
    CHECK_CUDA(cudaMalloc(&d_input2, size));
    CHECK_CUDA(cudaMalloc(&d_output1, size));
    CHECK_CUDA(cudaMalloc(&d_output2, size));
    CHECK_CUDA(cudaMalloc(&d_final_out, size));
    CHECK_CUDA(cudaMalloc(&d_weights, LUT_SIZE * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input1, h_input1, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input2, h_input2, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, LUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Create 3 streams
    cudaStream_t stream1, stream2, stream3;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));
    CHECK_CUDA(cudaStreamCreate(&stream3));

    cudaEvent_t event1, event2, event3, forkEvent;
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventCreate(&event2));
    CHECK_CUDA(cudaEventCreate(&event3));
    CHECK_CUDA(cudaEventCreate(&forkEvent));

    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));

    const int NUM_WARMUP = 5;
    const int NUM_ITERS = 20;

    // ========================================================================
    // Baseline: WITHOUT PDL (Sequential)
    // ========================================================================
    printf("=== Baseline: WITHOUT PDL ===\n");

    // Warmup
    for (int i = 0; i < NUM_WARMUP; i++) {
        producer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input1, d_output1, N);
        producer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input2, d_output2, N);
        consumerKernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_output1, d_output2, d_weights, d_final_out, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Benchmark
    CHECK_CUDA(cudaEventRecord(startEvent));
    for (int i = 0; i < NUM_ITERS; i++) {
        producer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input1, d_output1, N);
        producer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input2, d_output2, N);
        consumerKernelNoPDL<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(
            d_output1, d_output2, d_weights, d_final_out, N);
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaEventSynchronize(stopEvent));

    float timeNoPDL = 0;
    CHECK_CUDA(cudaEventElapsedTime(&timeNoPDL, startEvent, stopEvent));

    printf("  Total time: %.3f ms\n", timeNoPDL);
    printf("  Average per iteration: %.3f ms\n", timeNoPDL / NUM_ITERS);
    printf("  Pattern: Producer1 → Producer2 → Consumer (sequential)\n\n");

    // ========================================================================
    // WITH PDL: 2 Producers → 1 Consumer
    // ========================================================================
    printf("=== WITH PDL: 2 Producers → 1 Consumer ===\n");

    printf("Capturing graph with PDL edges...\n");

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    // Fork to stream2 for parallel producers
    CHECK_CUDA(cudaEventRecord(forkEvent, stream1));
    CHECK_CUDA(cudaStreamWaitEvent(stream2, forkEvent, 0));

    // Producer1 on stream1, Producer2 on stream2 (run in parallel)
    producer1Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_input1, d_output1, N);
    producer2Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_input2, d_output2, N);

    // Join pattern: Both producers must complete before consumer
    CHECK_CUDA(cudaEventRecord(event1, stream1));
    CHECK_CUDA(cudaEventRecord(event2, stream2));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event1, 0));
    CHECK_CUDA(cudaStreamWaitEvent(stream3, event2, 0));

    // Consumer on stream3 (depends on both producers)
    consumerKernel<<<blocksPerGrid, threadsPerBlock, 0, stream3>>>(
        d_output1, d_output2, d_weights, d_final_out, N);

    // Join back to stream1
    CHECK_CUDA(cudaEventRecord(event3, stream3));
    CHECK_CUDA(cudaStreamWaitEvent(stream1, event3, 0));

    CHECK_CUDA(cudaStreamEndCapture(stream1, &graph));
    printf("✓ Graph captured\n");

    // Configure PDL edges
    printf("Configuring PDL edges...\n");

    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    cudaGraphNode_t *nodes = (cudaGraphNode_t*)malloc(numNodes * sizeof(cudaGraphNode_t));
    CHECK_CUDA(cudaGraphGetNodes(graph, nodes, &numNodes));

    cudaGraphNode_t producer1Node = NULL, producer2Node = NULL, consumerNode = NULL;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        CHECK_CUDA(cudaGraphNodeGetType(nodes[i], &nodeType));
        if (nodeType == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params;
            CHECK_CUDA(cudaGraphKernelNodeGetParams(nodes[i], &params));
            if (params.func == (void*)producer1Kernel) producer1Node = nodes[i];
            else if (params.func == (void*)producer2Kernel) producer2Node = nodes[i];
            else if (params.func == (void*)consumerKernel) consumerNode = nodes[i];
        }
    }

    if (!producer1Node || !producer2Node || !consumerNode) {
        fprintf(stderr, "Failed to find kernel nodes\n");
        exit(EXIT_FAILURE);
    }

    // Remove default edges
    cudaGraphNode_t fromRemove[] = {producer1Node, producer2Node};
    cudaGraphNode_t toRemove[] = {consumerNode, consumerNode};
    CHECK_CUDA(cudaGraphRemoveDependencies(graph, fromRemove, toRemove, NULL, 2));

    // Add TWO PDL edges: Producer1 → Consumer, Producer2 → Consumer
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
    cudaGraphNode_t toAdd[] = {consumerNode, consumerNode};
    cudaGraphEdgeData edgeArray[] = {edgeData1, edgeData2};

    CHECK_CUDA(cudaGraphAddDependencies(graph, fromAdd, toAdd, edgeArray, 2));
    printf("✓ PDL edges configured:\n");
    printf("  Producer1 → Consumer (PDL)\n");
    printf("  Producer2 → Consumer (PDL)\n");
    printf("  Consumer calls cudaGridDependencySynchronize() to wait for BOTH!\n\n");

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
    printf("  Pattern: Producer1 & Producer2 trigger → Consumer launches\n");
    printf("           Consumer does independent work (LUT loading)\n");
    printf("           cudaGridDependencySynchronize() waits for BOTH producers\n\n");

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
        printf("  Consumer launches when first producer triggers (50%% done)\n");
        printf("  Consumer loads LUT while waiting for both producers\n");
        printf("  cudaGridDependencySynchronize() ensures both producer outputs ready!\n");
    } else {
        printf("\n  Note: PDL benefit may be small on this workload\n");
    }

    printf("\n=== Profile with Nsight Systems ===\n");
    printf("nsys profile -t cuda --cuda-graph-trace=node \\\n");
    printf("             -o pdl_two_producers ./cuda_pdl_two_producers\n\n");
    printf("Expected timeline:\n");
    printf("  Producer1 (35μs) ────────────────┐\n");
    printf("  Producer2 (35μs) ────────────────┤\n");
    printf("              50%% ↓                 │\n");
    printf("            Consumer loads LUT ─────┤ waits for BOTH\n");
    printf("                                    ↓\n");
    printf("                    cudaGridDependencySynchronize()\n");
    printf("                                    ↓\n");
    printf("                    Consumer combines outputs\n");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaEventDestroy(forkEvent));
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaEventDestroy(event2));
    CHECK_CUDA(cudaEventDestroy(event3));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaStreamDestroy(stream3));
    CHECK_CUDA(cudaFree(d_input1));
    CHECK_CUDA(cudaFree(d_input2));
    CHECK_CUDA(cudaFree(d_output1));
    CHECK_CUDA(cudaFree(d_output2));
    CHECK_CUDA(cudaFree(d_final_out));
    CHECK_CUDA(cudaFree(d_weights));
    free(h_input1);
    free(h_input2);
    free(h_weights);

    return 0;
}
