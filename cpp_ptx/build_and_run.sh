#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_CUTLASS_DIR="/mnt/c/Users/dibak/Desktop/github_repos/cutlass"
CUTLASS_DIR="${CUTLASS_DIR:-${DEFAULT_CUTLASS_DIR}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
NCU="${NCU:-$(command -v ncu || true)}"

SM_ARCH="${SM_ARCH:-80}"
CXX_STD="${CXX_STD:-c++17}"
RUN_AFTER_BUILD=1
PTX_ONLY=0
VERBOSE=0
PROGRAM_ARGS=()
RUN_WITH_NCU=0
NCU_ARGS=()
NCU_CHECK_EXIT_CODE=1
KERNEL_NAME=""
KERNEL_DIR=""
SOURCE_FILE=""
HEADER_FILE=""
BUILD_DIR=""
OUTPUT_BIN=""
OUTPUT_PTX=""
NCU_OUTPUT=""

usage() {
    cat <<EOF
Usage: $(basename "$0") <kernel> [options] [-- program_args...]
       $(basename "$0") --list

Compile a kernel driver from cpp_ptx/<kernel>, generate PTX, and optionally run it.

The kernel directory must contain:
  - <kernel>.cu and <kernel>.h, or
  - exactly one .cu file and exactly one .h file

Options:
  --list              List available kernels under ${SCRIPT_DIR}
  --sm <arch>         Target SM architecture (default: ${SM_ARCH})
  --cutlass-dir <dir> CUTLASS root containing include/cute/tensor.hpp
  --build-dir <dir>   Output directory (default: cpp_ptx/<kernel>/build)
  --compile-only      Build the executable and PTX, but do not run
  --ptx-only          Generate only the PTX file
  --ncu               Run the driver under Nsight Compute
  --ncu-output <path> Base path for the Nsight Compute report
  --ncu-arg <arg>     Extra argument to forward to ncu (repeatable)
  --ncu-ignore-exit   Tell ncu not to fail the profile when the target app exits non-zero
  --verbose           Print commands before running them
  --                  Pass the remaining arguments to the driver executable
  -h, --help          Show this message

Environment overrides:
  NVCC, NCU, CUDA_HOME, CUTLASS_DIR, SM_ARCH, CXX_STD
EOF
}

log() {
    printf '[cpp-ptx] %s\n' "$*"
}

die() {
    printf '[cpp-ptx] ERROR: %s\n' "$*" >&2
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

list_kernels() {
    local found=0
    local dir

    for dir in "${SCRIPT_DIR}"/*; do
        [[ -d "${dir}" ]] || continue

        if compgen -G "${dir}/*.cu" > /dev/null && compgen -G "${dir}/*.h" > /dev/null; then
            printf '%s\n' "$(basename "${dir}")"
            found=1
        fi
    done

    if [[ "${found}" -eq 0 ]]; then
        log "No kernel directories with both .cu and .h files were found under ${SCRIPT_DIR}"
    fi
}

resolve_single_file() {
    local pattern="$1"
    local description="$2"
    local -n result_ref="$3"
    local matches=()

    while IFS= read -r match; do
        matches+=("${match}")
    done < <(compgen -G "${pattern}" || true)

    if [[ "${#matches[@]}" -eq 1 ]]; then
        result_ref="${matches[0]}"
        return 0
    fi

    if [[ "${#matches[@]}" -eq 0 ]]; then
        die "Kernel '${KERNEL_NAME}' is missing a ${description} in ${KERNEL_DIR}"
    fi

    die "Kernel '${KERNEL_NAME}' has multiple ${description} files in ${KERNEL_DIR}; use a single pair or name them ${KERNEL_NAME}.cu and ${KERNEL_NAME}.h"
}

resolve_kernel_files() {
    KERNEL_DIR="${SCRIPT_DIR}/${KERNEL_NAME}"
    [[ -d "${KERNEL_DIR}" ]] || die "Kernel directory not found: ${KERNEL_DIR}"

    SOURCE_FILE="${KERNEL_DIR}/${KERNEL_NAME}.cu"
    HEADER_FILE="${KERNEL_DIR}/${KERNEL_NAME}.h"

    if [[ -f "${SOURCE_FILE}" && -f "${HEADER_FILE}" ]]; then
        return 0
    fi

    resolve_single_file "${KERNEL_DIR}/*.cu" ".cu source file" SOURCE_FILE
    resolve_single_file "${KERNEL_DIR}/*.h" ".h header file" HEADER_FILE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list)
            list_kernels
            exit 0
            ;;
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
        --ncu)
            RUN_WITH_NCU=1
            shift
            ;;
        --ncu-output)
            [[ $# -ge 2 ]] || die "--ncu-output requires a value"
            NCU_OUTPUT="$2"
            shift 2
            ;;
        --ncu-arg)
            [[ $# -ge 2 ]] || die "--ncu-arg requires a value"
            NCU_ARGS+=("$2")
            shift 2
            ;;
        --ncu-ignore-exit)
            NCU_CHECK_EXIT_CODE=0
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
        --)
            shift
            PROGRAM_ARGS=("$@")
            break
            ;;
        -*)
            die "Unknown argument: $1"
            ;;
        *)
            if [[ -n "${KERNEL_NAME}" ]]; then
                die "Kernel already set to '${KERNEL_NAME}'. Pass only one kernel name."
            fi
            KERNEL_NAME="$1"
            shift
            ;;
    esac
done

[[ -n "${KERNEL_NAME}" ]] || die "Missing kernel name. Use --list to see available kernels."

resolve_kernel_files

[[ -x "${NVCC}" ]] || die "nvcc not found or not executable: ${NVCC}"
[[ -d "${CUTLASS_DIR}" ]] || die "CUTLASS_DIR does not exist: ${CUTLASS_DIR}"
[[ -f "${CUTLASS_DIR}/include/cute/tensor.hpp" ]] || die \
    "Expected CuTe header at ${CUTLASS_DIR}/include/cute/tensor.hpp"

if [[ -z "${BUILD_DIR}" ]]; then
    BUILD_DIR="${KERNEL_DIR}/build"
fi

OUTPUT_BIN="${BUILD_DIR}/${KERNEL_NAME}"
OUTPUT_PTX="${BUILD_DIR}/${KERNEL_NAME}.ptx"

if [[ -z "${NCU_OUTPUT}" ]]; then
    NCU_OUTPUT="${BUILD_DIR}/${KERNEL_NAME}_ncu"
fi

if [[ "${RUN_WITH_NCU}" -eq 1 ]]; then
    [[ -n "${NCU}" ]] || die "ncu not found. Set NCU or add Nsight Compute to PATH."
    [[ -x "${NCU}" ]] || die "ncu is not executable: ${NCU}"
fi

mkdir -p "${BUILD_DIR}"

COMMON_FLAGS=(
    "-std=${CXX_STD}"
    "-O3"
    "-lineinfo"
    "-I${CUTLASS_DIR}/include"
    "-arch=sm_${SM_ARCH}"
)

log "Building kernel '${KERNEL_NAME}' for sm_${SM_ARCH}"
log "Source: ${SOURCE_FILE}"
log "Header: ${HEADER_FILE}"
log "Using CUTLASS from ${CUTLASS_DIR}"

if [[ "${PTX_ONLY}" -eq 0 ]]; then
    run_cmd "${NVCC}" "${COMMON_FLAGS[@]}" "${SOURCE_FILE}" -o "${OUTPUT_BIN}"
    log "Built executable: ${OUTPUT_BIN}"
fi

run_cmd "${NVCC}" "${COMMON_FLAGS[@]}" -ptx "${SOURCE_FILE}" -o "${OUTPUT_PTX}"
log "Generated PTX: ${OUTPUT_PTX}"

if [[ "${RUN_AFTER_BUILD}" -eq 1 ]]; then
    if [[ "${RUN_WITH_NCU}" -eq 1 ]]; then
        log "Running kernel '${KERNEL_NAME}' under Nsight Compute"
        NCU_CMD=(
            "${NCU}"
            "--mode"
            "launch-and-attach"
            "--target-processes"
            "application-only"
            "--profile-from-start"
            "yes"
            "--export"
            "${NCU_OUTPUT}"
            "--force-overwrite"
            "--check-exit-code"
            "${NCU_CHECK_EXIT_CODE}"
            "${NCU_ARGS[@]}"
            "${OUTPUT_BIN}"
            "${PROGRAM_ARGS[@]}"
        )
        if ! run_cmd "${NCU_CMD[@]}"; then
            die "Nsight Compute profiling failed. Re-run with --verbose to inspect the full ncu command, or try --ncu-ignore-exit if you want ncu to ignore the target application's exit code."
        fi
        log "Nsight Compute report written to ${NCU_OUTPUT}.ncu-rep"
    else
        log "Running kernel '${KERNEL_NAME}'"
        if ! run_cmd "${OUTPUT_BIN}" "${PROGRAM_ARGS[@]}"; then
            die "Kernel execution failed. The executable was built successfully, so check GPU visibility and that your installed NVIDIA driver supports the CUDA runtime used by ${NVCC}."
        fi
    fi
fi
