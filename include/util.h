#pragma once
#include "config.h"
#include <iostream>

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

#define DEBUG_PRINTS

#ifdef DEBUG_PRINTS
#define DEBUG_VEC(vec, n)                                                      \
  {                                                                            \
    printf(#vec " = ");                                                        \
    for (int i = 0; i < n; i++) {                                              \
      printf("%f ", vec(i));                                                   \
    }                                                                          \
    printf("\n");                                                              \
  }
#define DEBUG_MAT(vec, m, n)                                                   \
  {                                                                            \
    printf(#vec " = ");                                                        \
    for (int i = 0; i < n; i++) {                                              \
      for (int j = 0; j < m; j++) {                                            \
        printf("%f ", vec(i, j));                                              \
      }                                                                        \
      printf("\n");                                                            \
    }                                                                          \
    printf("\n");                                                              \
  }
#define TRACE(val)                                                             \
  { printf(__FILE__ ":%d - %s: " #val "\n", __LINE__, __PRETTY_FUNCTION__); }
#define DEBUG_FLOAT(float)                                                     \
  { printf(#float " = %f\n", float); }
#else
#define DEBUG_VEC(vec, n)                                                      \
  {}
#define DEBUG_MAT(vec, m, n)                                                   \
  {}
#define TRACE(val)                                                             \
  {}
#define DEBUG_FLOAT(float)                                                     \
  {}
#endif

using byte = unsigned char;

void *alloc_device_bytes(size_t bytes);
template <typename T> T *alloc_device(size_t count) {
  size_t bytes = count * sizeof(T);
  return reinterpret_cast<T *>(alloc_device_bytes(bytes));
}

__host__ __device__ void
memcpy_device_bytes(void *__restrict dest, void *__restrict src, size_t bytes);
template <typename T>
__host__ __device__ void memcpy_device(T *__restrict dest, T *__restrict src,
                                       size_t count) {
  memcpy_device_bytes(dest, src, count * sizeof(T));
}
void memcpy_host_device(void *__restrict dest, void *__restrict src,
                        size_t bytes);

template <typename T> T *move_array_to_device(T *host_ptr, size_t count) {
  auto new_ptr = alloc_device<T>(count);
  memcpy_host_device(new_ptr, host_ptr, count * sizeof(T));
  delete[] host_ptr;
  return new_ptr;
}
