# NVIDIA GPU Specifications

| GPU | Generation | Clock speed | SMs/chip | SMEM capacity/SM | L2 capacity/chip | HBM capacity/chip | HBM bandwidth/chip | Notes |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
| V100 | Volta | 1.25 GHz / 1.38 GHz | 80 | 96 KB | 6 MB | 32 GB | - | - |
| A100 | Ampere | 1.10 GHz / 1.41 GHz | 108 | 192 KB | 40 MB | 80 GB | - | - |
| H100 | Hopper | 1.59 GHz / 1.98 GHz | 132 | 256 KB | 50 MB | 80 GB | - | - |
| H200 | Hopper | 1.59 GHz / 1.98 GHz | 132 | 256 KB | 50 MB | 141 GB | - | - |
| B200 | Blackwell | Not published | 148 | 256 KB | 126 MB | 192 GB | - | - |
| Rubin GPU | Vera Rubin | Not published | 224 | Not published | Not published | 288 GB HBM4 | Up to 22 TB/s | NVIDIA lists 50 PFLOPS NVFP4 inference, 35 PFLOPS NVFP4 training, 17.5 PFLOPS FP8/FP6 training, 3.6 TB/s NVLink bandwidth|

## Vera Rubin Notes

- Vera Rubin NVL72 is a rack-scale platform with 72 Rubin GPUs and 36 Vera CPUs.
- One Vera Rubin Superchip contains 2 Rubin GPUs and 1 Vera CPU.
- NVIDIA marks the Vera Rubin NVL72 specifications as preliminary and subject to change.
- NVIDIA's Rubin architecture details published so far do not include clock speed, L2 cache capacity, or SMEM capacity per SM.

## Sources

- [NVIDIA Vera Rubin NVL72 specs](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/)
- [NVIDIA Technical Blog: Inside the NVIDIA Vera Rubin Platform](https://developer.nvidia.com/blog/inside-the-nvidia-rubin-platform-six-new-chips-one-ai-supercomputer/)
- [NVIDIA Newsroom: Vera Rubin Opens Agentic AI Frontier](https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform)
