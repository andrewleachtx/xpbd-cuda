#pragma once
#include "config.h"

#define CUDA_CHECK(expr)                                                       \
  {                                                                            \
    cudaError_t err = (expr);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr,                                                          \
              __FILE__                                                         \
              "(%d):\n CUDA Error Code  : %d\n     Error String: %s\n",        \
              __LINE__, err, cudaGetErrorString(err));                         \
      exit(err);                                                               \
    }                                                                          \
  }

void *alloc_device(size_t bytes);

void memcpy_device(void *__restrict dest, void *__restrict src, size_t bytes);
