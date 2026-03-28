#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_FILE="${SCRIPT_DIR}/swizzled_transpose.cu"
BUILD_DIR="${SCRIPT_DIR}/build"
OUTPUT_BIN="${BUILD_DIR}/swizzled_transpose"
OUTPUT_PTX="${BUILD_DIR}/swizzled_transpose.ptx"

DEFAULT_CUTLASS_DIR="/mnt/c/Users/dibak/Desktop/github_repos/cutlass"
CUTLASS_DIR="${CUTLASS_DIR:-${DEFAULT_CUTLASS_DIR}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"

SM_ARCH="${SM_ARCH:-80}"
CXX_STD="${CXX_STD:-c++17}"
RUN_AFTER_BUILD=1
PTX_ONLY=0
VERBOSE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

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
EOF
}

log() {
    printf '[swizzled-transpose] %s\n' "$*"
}

die() {
    printf '[swizzled-transpose] ERROR: %s\n' "$*" >&2
    exit 1
}

run_cmd() {
    if [[ "${VERBOSE}" -eq 1 ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    fi
    "$@"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sm)
            [[ $# -ge 2 ]] || die "--sm requires a value"
            SM_ARCH="$2"
            shift 2
            ;;
        --cutlass-dir)
            [[ $# -ge 2 ]] || die "--cutlass-dir requires a value"
            CUTLASS_DIR="$2"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || die "--build-dir requires a value"
            BUILD_DIR="$2"
            OUTPUT_BIN="${BUILD_DIR}/swizzled_transpose"
            OUTPUT_PTX="${BUILD_DIR}/swizzled_transpose.ptx"
            shift 2
            ;;
        --compile-only)
            RUN_AFTER_BUILD=0
            shift
            ;;
        --ptx-only)
            PTX_ONLY=1
            RUN_AFTER_BUILD=0
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

[[ -f "${SOURCE_FILE}" ]] || die "Missing source file: ${SOURCE_FILE}"
[[ -x "${NVCC}" ]] || die "nvcc not found or not executable: ${NVCC}"
[[ -d "${CUTLASS_DIR}" ]] || die "CUTLASS_DIR does not exist: ${CUTLASS_DIR}"
[[ -f "${CUTLASS_DIR}/include/cute/tensor.hpp" ]] || die \
    "Expected CuTe header at ${CUTLASS_DIR}/include/cute/tensor.hpp"

mkdir -p "${BUILD_DIR}"

COMMON_FLAGS=(
    "-std=${CXX_STD}"
    "-O3"
    "-lineinfo"
    "-I${CUTLASS_DIR}/include"
    "-arch=sm_${SM_ARCH}"
)

log "Building for sm_${SM_ARCH}"
log "Using CUTLASS from ${CUTLASS_DIR}"

if [[ "${PTX_ONLY}" -eq 0 ]]; then
    run_cmd "${NVCC}" "${COMMON_FLAGS[@]}" "${SOURCE_FILE}" -o "${OUTPUT_BIN}"
    log "Built executable: ${OUTPUT_BIN}"
fi

run_cmd "${NVCC}" "${COMMON_FLAGS[@]}" -ptx "${SOURCE_FILE}" -o "${OUTPUT_PTX}"
log "Generated PTX: ${OUTPUT_PTX}"

if [[ "${RUN_AFTER_BUILD}" -eq 1 ]]; then
    log "Running driver"
    if ! run_cmd "${OUTPUT_BIN}"; then
        die "Driver execution failed. The executable was built successfully, so check GPU visibility and that your installed NVIDIA driver supports the CUDA runtime used by ${NVCC}."
    fi
fi
