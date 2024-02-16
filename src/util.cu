#include "util.h"

void *alloc_device_bytes(size_t bytes) {
#ifdef USE_CUDA
  void *device_ptr;
  CUDA_CHECK(cudaMalloc(&device_ptr, bytes));
  return device_ptr;
#else
  return malloc(bytes);
#endif
}

void memcpy_device(void *__restrict dest, void *__restrict src, size_t bytes) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyDeviceToDevice));
#else
  memcpy(dest, src, bytes);
#endif
}

void memcpy_host_device(void *__restrict dest, void *__restrict src,
                        size_t bytes) {
#ifdef USE_CUDA
  CUDA_CHECK(cudaMemcpy(dest, src, bytes, cudaMemcpyHostToDevice));
#else
  memcpy(dest, src, bytes);
#endif
}
