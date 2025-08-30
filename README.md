# cuda-kernels (CUDA Week 2)

Small repo to learn CUDA basics by writing and benchmarking a few kernels.

## What’s here
- `vec_add.cu` – 1D vector add (one thread per element)
- `matmul.cu` – matrix multiply
  - `matmul_naive` (global memory only)
  - `matmul_tiled` (shared-memory tiling)
- `main.cu` – tiny harness to time kernels with CUDA events and dump `bench/results.csv`
- `bench/plot.py` – quick matplotlib plot of tile size vs GFLOPs (optional)

## Requirements
- NVIDIA GPU (tested on RTX 3060)
- CUDA Toolkit 12.x
- Driver working: `nvidia-smi`
- Compiler: `nvcc` (C++17 is fine)

## Build
### Quick (nvcc)
```bash
nvcc -O3 -arch=sm_86 -o kernels main.cu matmul.cu vec_add.cu
```

### CMake (optional)
```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
```

## Run
```bash
./kernels
```
This runs warmups, times each kernel, prints best time, and appends to `bench/results.csv`:
```
algo,M,N,K,ms,gflops
naive,512,512,512,XX.XXX,YY.YY
tiled,512,512,512,AA.AAA,BB.BB
```

Change matrix sizes in `main.cu` (defaults: 512×512×512).

## Plot (optional)
```bash
python3 bench/plot.py
```
Produces `tile_vs_gflops.png`.

## Notes
- **Global vs Shared memory**: global is large but high latency; shared is on‑chip and fast. Tiling loads blocks of A and B into shared memory, so each value is reused by many threads—fewer global reads → big speedup.
- **Coalescing**: arrange loads so threads in a warp read neighboring addresses. The tiled kernel does this for both A and B tiles.
- **Timing**: we time kernel execution with CUDA events (no H2D/D2H).

## Sanity checks
- Compare against a small CPU matmul once (e.g., 128³) to verify correctness.
- Quick profiler:
```bash
ncu --section LaunchStats --section Occupancy ./kernels
```
Look at *Achieved Occupancy* and DRAM/shared throughput.

---
Keep it simple. Get it correct. Then make it fast.
