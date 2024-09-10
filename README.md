# XPBD CUDA

A CUDA-accelerated XPBD-based physics simulation framework.

## What I Needed To Install
1. Visual Studio (17, 2022) https://visualstudio.microsoft.com/
2. CUDA (12.6) https://developer.nvidia.com/cuda-downloads
3. Download / clone Eigen into `include/` and follow the "INSTALL" file (create `build/` in `eigen3.x.x/`, cd into, cmake ..)
4. The first time I did `cmake -S . -B build` in `xpbd-cuda/` I couldn't use cmd/pwsh, I had to use "x64 Native Tools Command Prompt for VS 2022" because otherwise it wouldn't acknowledge my CUDA. Probably other workarounds but this was fastest.
5. Trying to build/compile in VS came with a LOT of errors - guessing version specific to my CUDA or Eigen3 usage. Not sure.

## Building

```bash
cmake -S . -B build
# or with options
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=ON -DWRITE=OFF
cmake --build build
```

## Running Tests

```bash
cmake --build build --target <test>
```

## Profiling or Benchmarking

See `run_perf_test.sh` and `run_profiler.sh`.

## Formatting

```bash
git ls-files -- '*.cu' '*.h' | xargs clang-format -i -style=file
```

## Acknowledgments

Code in the JGT-float modules is public domain and accessed from this URL:

[https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/](https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/)

It has been edited to be CUDA-compatible.

