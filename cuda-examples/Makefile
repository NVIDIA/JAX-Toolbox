# Makefile for CUDA Graph PDL with Multi-Stream Execution

# CUDA compiler
NVCC = nvcc

# Target executable (stream capture with multi-stream + PDL)
TARGET = cuda_graph_pdl_stream_capture

# Source file
SRC = cuda_graph_pdl_stream_capture.cu

# CUDA compute capability (adjust for your GPU)
# SM 10.0 for Blackwell (GB200)
# SM 9.0 for Hopper (H100)
# SM 8.0 for Ampere (A100)
CUDA_ARCH = sm_90

# Compiler flags
NVCC_FLAGS = -arch=$(CUDA_ARCH) -O3

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SRC)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Generate Nsight Systems profile report
nsys-profile: $(TARGET)
	nsys profile -o nsys_report_$(TARGET) --stats=true ./$(TARGET)

# Generate Nsight Compute profile report
ncu-profile: $(TARGET)
	ncu -o ncu_report_$(TARGET) --set full ./$(TARGET)

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -f nsys_report_$(TARGET).* ncu_report_$(TARGET).*

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
	@echo ""
	@echo "Targets:"
	@echo "  all (default)  - Build the executable"
	@echo "  run            - Build and run the program"
	@echo "  nsys-profile   - Generate Nsight Systems profile report"
	@echo "  ncu-profile    - Generate Nsight Compute profile report"
	@echo "  clean          - Remove build artifacts and profile reports"
	@echo "  version        - Display CUDA version"
	@echo "  gpu-info       - Display GPU information"
	@echo "  help           - Display this help message"
	@echo ""
	@echo "Requirements:"
	@echo "  - CUDA 12.0+ for PDL support"
	@echo "  - SM 9.0+ GPU (Hopper, Blackwell) for full PDL features"
	@echo "  - Adjust CUDA_ARCH variable for your GPU architecture"

.PHONY: all run nsys-profile ncu-profile clean version gpu-info help
