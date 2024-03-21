#pragma once
#include <stddef.h>

namespace data {

struct _SOAStoreBodyRigid {
  float *x0;
  float *x1;
  float *x2;
};

class SOAStore {
public:
  struct _SOAStoreBodyRigid body_rigid;

  SOAStore();
  SOAStore(size_t body_rigid_count);

  void deallocate();

  template <typename T>
  constexpr __host__ __device__ size_t get_aligned_size(size_t count) {
    const size_t alignment_count = 32;
    const size_t byte_size = count * sizeof(T);
    if (byte_size % alignment_count == 0)
      return byte_size;
    else
      return (byte_size / alignment_count + 1) * alignment_count;
  }
};

#ifdef USE_CUDA
extern __device__ SOAStore global_store;
#else
extern SOAStore global_store;
#endif

} // namespace data
