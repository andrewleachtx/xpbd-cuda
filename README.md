# PBD Rigidbody CUDA

A CUDA-accelerated Differential XPBD Framework

## Building

```bash
cmake -S . -B build
cmake --build build
```

## Running Tests

```bash
cmake --build build --target test
```

## Formatting

```bash
git ls-files -- '*.cu' '*.h' | xargs clang-format -i -style=file
```

## Acknowledgements

Code in the JGT-float modules is public domain and accessed from this URL:

[https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/](https://web.archive.org/web/20070715170639/jgt.akpeters.com/papers/MahovskyWyvill04/)

It has been edited to be CUDA-compatible.
