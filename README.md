# XPBD CUDA

A CUDA-accelerated XPBD-based physics simulation framework.

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

