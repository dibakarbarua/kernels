# `cpp_ptx`

This directory holds standalone CUDA C++ kernel drivers that can be compiled to both native executables and PTX.

Each kernel lives in its own subdirectory under `cpp_ptx/`. The shared [`build_and_run.sh`](/mnt/c/Users/dibak/Desktop/github_repos/cutedsl_kernels/cpp_ptx/build_and_run.sh) script builds any kernel directory that contains:

- `<kernel>/<kernel>.cu` and `<kernel>/<kernel>.h`, or
- exactly one `.cu` file and exactly one `.h` file in that directory

Build artifacts stay local to each kernel in `<kernel>/build/`.

## Layout

```text
cpp_ptx/
├── build_and_run.sh
├── README.md
└── <kernel>/
    ├── <kernel>.cu
    ├── <kernel>.h
    └── build/
```

## Usage

List available kernels:

```bash
./build_and_run.sh --list
```

Build and run a kernel:

```bash
./build_and_run.sh swizzled_transpose
```

Build only:

```bash
./build_and_run.sh swizzled_transpose --compile-only
```

Generate PTX only:

```bash
./build_and_run.sh swizzled_transpose --ptx-only
```

Target a different SM architecture:

```bash
./build_and_run.sh swizzled_transpose --sm 90
```

Pass arguments through to the compiled executable:

```bash
./build_and_run.sh swizzled_transpose -- --example-arg value
```

Run under Nsight Compute:

```bash
./build_and_run.sh swizzled_transpose --ncu
```

## Script behavior

The script:

- validates that the selected kernel directory exists
- locates a matching `.cu` and `.h`
- compiles the `.cu` file with `nvcc`
- emits a PTX file alongside the executable in that kernel's `build/` directory
- can optionally launch the executable directly or under `ncu`

Supported overrides:

- `CUTLASS_DIR`
- `CUDA_HOME`
- `NVCC`
- `NCU`
- `SM_ARCH`
- `CXX_STD`
