#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <inttypes.h>
#include <chrono>
#include <limits>
#include <tuple>


#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: CUDA error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed: NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


static int getEnvAsInt(const char* name, int default_val) {
  char* str_val = getenv(name);
  if (!str_val) {
    return default_val;
  }
  int int_val;
  if (sscanf(str_val, "%d", &int_val) != 1) {
    printf("Failed: Could not parse env var %s as int: '%s'\n", name, str_val);
    exit(EXIT_FAILURE);
  }
  return int_val;
}


void printUsageAndAbort(char* progName) {
  printf("Usage: %s world-size world-rank local-rank coordinator-address\n", progName);
  exit(EXIT_FAILURE);
}


void parseArgs(int argc, char* argv[], int* nRanks, int* myRank, int* localRank,
    char* coordinatorAddress) {
  if (argc != 5) {
    printUsageAndAbort(argv[0]);
  }
  if (sscanf(argv[1], "%d", nRanks) != 1 || *nRanks <= 0) {
    printf("Expected world-size to be a positive integer\n");
    printUsageAndAbort(argv[0]);
  }
  if (sscanf(argv[2], "%d", myRank) != 1 || *myRank < 0 || *myRank >= *nRanks) {
    printf("Expected world-rank to be an integer in [0;world-size)\n");
    printUsageAndAbort(argv[0]);
  }
  if (sscanf(argv[3], "%d", localRank) != 1 || *localRank < 0) {
    printf("Expected local-rank to be a non-negative integer\n");
    printUsageAndAbort(argv[0]);
  }
  if (sscanf(argv[4], "%127s", coordinatorAddress) != 1 ||
      strlen(coordinatorAddress) >= 127) {
    printf("Expected coordinator-address to be a string (ip:port)\n");
    printUsageAndAbort(argv[0]);
  }
}


std::tuple<uint64_t, uint64_t> sampleAllReduces(int rank, int nRanks, ncclUniqueId id,
    int size, int rounds) {
  float *sendbuff, *recvbuff;
  CUDACHECK(cudaMalloc((void**) &sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc((void**) &recvbuff, size * sizeof(float)));
  
  cudaStream_t s;
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, rank));

  // Sample a few rounds of minimal all-reduces
  uint64_t minDuration = std::numeric_limits<uint64_t>::max();
  uint64_t maxDuration = 0;
  for (int i=0; i<rounds; i++) {
    auto t_start = std::chrono::high_resolution_clock::now();

    NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size, ncclFloat,
      ncclSum, comm, s));
    CUDACHECK(cudaStreamSynchronize(s));

    uint64_t duration = std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::high_resolution_clock::now() - t_start).count();
    if (duration < minDuration) {
      minDuration = duration;
    }
    if (duration > maxDuration) {
      maxDuration = duration;
    }
  }

  // Clean up
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));
  ncclCommDestroy(comm);

  return {minDuration, maxDuration};
}


int main(int argc, char* argv[])
{
  // Number of floats communicated in all-reduce
  int size = 1;
  // Number of all-reduces to sample, only best result is considered.
  int rounds = 10;
  // The minimum duration required to pass the sanity check.
  int threshold = getEnvAsInt("NCCL_SANITY_CHECK_LATENCY_US", 1000);

  int nRanks, myRank, localRank;
  char coordinatorAddress[128];
  parseArgs(argc, argv, &nRanks, &myRank, &localRank, coordinatorAddress);

  CUDACHECK(cudaSetDevice(localRank));

  // Compute same NCCL unique id in all ranks
  ncclUniqueId id;
  if (setenv("NCCL_COMM_ID", coordinatorAddress, 1) != 0) {
    printf("Failed: Could not set NCCL_COMM_ID\n");
    exit(EXIT_FAILURE);
  }

  // ncclUniqueId is just a ncclBootstrapHandle (with some padding), see:
  //   https://github.com/NVIDIA/nccl/blob/v2.19/src/include/bootstrap.h#L14
  // In the following call, the addr is initialized using NCCL_COMM_ID, but the
  // magic is drawn from urandom bits. Usually this would only be done on rank 0
  // and the resulting id would then be broadcast to all the other processes
  // out-of-band (e.g. using standard MPI). Instead, here we've already settled
  // on an appropriate ip and port for rank 0 (given in NCCL_COMMI_ID), and all
  // that remains is fixing the magic.
  ncclGetUniqueId(&id);
  *((uint64_t*) &id) = 0xDEADBEEFDEADBEEF;

  // Estimate latency by running several all-reduces
  auto[minDuration, maxDuration] = sampleAllReduces(myRank, nRanks, id, size, rounds);

  // Report result of the sanity check
  bool success = threshold >= minDuration;
  printf(
    "nccl-sanity-check success=%d rank=%d nRanks=%d rounds=%d threshold=%dus "
    "minDuration=%" PRIu64 "us maxDuration=%" PRIu64 "us\n",
    success, myRank, nRanks, rounds, threshold, minDuration, maxDuration);
  return !success;
}
