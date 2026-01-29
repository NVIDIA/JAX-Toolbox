# Makefile for CUDA Graph PDL with Multi-Stream Execution

# CUDA compiler
NVCC = nvcc

# Target executables
TARGET = cuda_graph_pdl_stream_capture
BENCHMARK_TARGET = cuda_graph_pdl_benchmark
IMPACT_TARGET = cuda_graph_pdl_impact

# Source files
SRC = cuda_graph_pdl_stream_capture.cu
BENCHMARK_SRC = cuda_graph_pdl_benchmark.cu
IMPACT_SRC = cuda_graph_pdl_impact.cu

# CUDA compute capability (adjust for your GPU)
# SM 10.0 for Blackwell (GB200)
# SM 9.0 for Hopper (H100)
# SM 8.0 for Ampere (A100)
CUDA_ARCH = sm_90

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3

# Default target - build all
all: $(TARGET) $(BENCHMARK_TARGET) $(IMPACT_TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Build benchmark target
$(BENCHMARK_TARGET): $(BENCHMARK_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(BENCHMARK_TARGET) $(BENCHMARK_SRC)

# Build impact demo target
$(IMPACT_TARGET): $(IMPACT_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(IMPACT_TARGET) $(IMPACT_SRC)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run the benchmark
benchmark-run: $(BENCHMARK_TARGET)
	./$(BENCHMARK_TARGET)

# Run the PDL impact demo
impact-run: $(IMPACT_TARGET)
	./$(IMPACT_TARGET)

# Generate Nsight Systems profile report
nsys-profile: $(TARGET)
	nsys profile -o nsys_report_$(TARGET) --stats=true ./$(TARGET)

# Generate Nsight Compute profile report
ncu-profile: $(TARGET)
	ncu -o ncu_report_$(TARGET) --set full ./$(TARGET)

# Generate Nsight Systems profile report for benchmark
benchmark-nsys-profile: $(BENCHMARK_TARGET)
	nsys profile -o nsys_report_$(BENCHMARK_TARGET) --stats=true ./$(BENCHMARK_TARGET)

# Generate Nsight Compute profile report for benchmark
benchmark-ncu-profile: $(BENCHMARK_TARGET)
	ncu -o ncu_report_$(BENCHMARK_TARGET) --set full ./$(BENCHMARK_TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(BENCHMARK_TARGET) $(IMPACT_TARGET)
	rm -f nsys_report_$(TARGET).* ncu_report_$(TARGET).*
	rm -f nsys_report_$(BENCHMARK_TARGET).* ncu_report_$(BENCHMARK_TARGET).*
	rm -f nsys_report_$(IMPACT_TARGET).* ncu_report_$(IMPACT_TARGET).*

# Display CUDA version
version:
	@nvcc --version

# Display GPU info
gpu-info:
	@nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Help
help:
	@echo "Makefile for CUDA Graph PDL with Multi-Stream Execution"
	@echo ""
	@echo "Features:"
	@echo "  - Stream capture with <<<>>> syntax"
	@echo "  - True multi-stream execution (stream1, stream2)"
	@echo "  - Programmatic Dependent Launch (PDL) edges"
	@echo "  - Proper cross-stream synchronization"
	@echo "  - Performance benchmark comparing traditional vs graph launches"
	@echo ""
	@echo "Targets:"
	@echo "  all (default)           - Build all executables"
	@echo "  run                     - Build and run the main program"
	@echo "  benchmark-run           - Build and run the performance benchmark"
	@echo "  impact-run              - Build and run the PDL impact demonstration"
	@echo "  nsys-profile            - Generate Nsight Systems profile for main program"
	@echo "  ncu-profile             - Generate Nsight Compute profile for main program"
	@echo "  benchmark-nsys-profile  - Generate Nsight Systems profile for benchmark"
	@echo "  benchmark-ncu-profile   - Generate Nsight Compute profile for benchmark"
	@echo "  clean                   - Remove all build artifacts and profile reports"
	@echo "  version                 - Display CUDA version"
	@echo "  gpu-info                - Display GPU information"
	@echo "  help                    - Display this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA 12.0+ for PDL support"
	@echo "  - SM 9.0+ GPU (Hopper, Blackwell) for full PDL features"
	@echo "  - Adjust CUDA_ARCH variable for your GPU architecture"

.PHONY: all run benchmark-run impact-run nsys-profile ncu-profile benchmark-nsys-profile benchmark-ncu-profile clean version gpu-info help
