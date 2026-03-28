Compile the swizzled transpose driver, generate PTX, and optionally run it.

Options:
  --sm <arch>         Target SM architecture (default: ${SM_ARCH})
  --cutlass-dir <dir> CUTLASS root containing include/cute/tensor.hpp
  --build-dir <dir>   Output directory (default: ${BUILD_DIR})
  --compile-only      Build the executable and PTX, but do not run
  --ptx-only          Generate only the PTX file
  --verbose           Print nvcc commands before running them
  -h, --help          Show this message

Environment overrides:
  NVCC, CUDA_HOME, CUTLASS_DIR, SM_ARCH, CXX_STD

./build_and_run.sh --compile-only
./build_and_run.sh
./build_and_run.sh --sm 90
./build_and_run.sh --ptx-only