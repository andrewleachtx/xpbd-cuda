#pragma once
#include <stddef.h>
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "apbd/Body.h"
#include "apbd/Shape.h"
#include "util.h"
#include <Eigen/Dense>

namespace data {

__host__ __device__ size_t soa_index(unsigned int index);

template <typename T>
constexpr __host__ __device__ size_t get_aligned_size(size_t count) {
  const size_t alignment_count = 32;
  const size_t byte_size = count * sizeof(T);
  if (byte_size % alignment_count == 0)
    return byte_size;
  else
    return (byte_size / alignment_count + 1) * alignment_count;
}

template <typename T>
T *get_aligned_buffer_segment(byte *data_store, size_t &offset, size_t count) {
  T *buffer_segment = reinterpret_cast<T *>(data_store + offset);
  offset += get_aligned_size<T>(count);
  return buffer_segment;
}

typedef Eigen::Matrix<float, 7, 1> vec7;
typedef Eigen::Matrix<float, 12, 1> vec12;

template <typename T> struct _SOAStoreGeneric {
  T *data;

  __host__ __device__ _SOAStoreGeneric() {}
  _SOAStoreGeneric(byte *data_store, size_t &offset, size_t count);
  static constexpr size_t size(size_t count) {
    return get_aligned_size<T>(count);
  }

  __host__ __device__ T get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, const T &new_val);
};

template <typename T>
_SOAStoreGeneric<T>::_SOAStoreGeneric(byte *data_store, size_t &offset,
                                      size_t count) {
  this->data = get_aligned_buffer_segment<T>(data_store, offset, count);
}

template <typename T> T _SOAStoreGeneric<T>::get(unsigned int index) const {
  return data[soa_index(index)];
}

template <typename T>
void _SOAStoreGeneric<T>::set(unsigned int index, const T &new_val) {
  data[soa_index(index)] = new_val;
}

struct _SOAStoreQuaterion {
  float4 *data;

  __host__ __device__ _SOAStoreQuaterion() {}
  _SOAStoreQuaterion(byte *data_store, size_t &offset, size_t count);
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count);
  }

  __host__ __device__ Eigen::Quaternionf get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Quaternionf new_val);
};

struct _SOAStoreVec3 {
  float2 *x01;
  float *x2;

  __host__ __device__ _SOAStoreVec3() {}
  _SOAStoreVec3(byte *data_store, size_t &offset, size_t count);
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float2>(count) + get_aligned_size<float>(count);
  }

  __host__ __device__ Eigen::Vector3f get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, Eigen::Vector3f new_val);
};

struct _SOAStoreVec7 {
  float4 *x03;
  float2 *x45;
  float *x6;

  __host__ __device__ _SOAStoreVec7() {}
  _SOAStoreVec7(byte *data_store, size_t &offset, size_t count);
  static constexpr size_t size(size_t count) {
    return get_aligned_size<float4>(count) + get_aligned_size<float2>(count) +
           get_aligned_size<float>(count);
  }

  __host__ __device__ vec7 get(unsigned int index) const;
  __host__ __device__ void set(unsigned int index, vec7 new_val);
};

struct _SOAStoreBodyRigid {
  _SOAStoreVec7 xdotInit;
  _SOAStoreVec3 position;
  _SOAStoreQuaterion rotation;
  _SOAStoreVec7 x0;
  _SOAStoreVec7 x1;
  _SOAStoreQuaterion x1_0_rot;
  _SOAStoreVec7 dxJacobi;
  _SOAStoreVec7 dxJacobiShock;
  _SOAStoreGeneric<bool> collide;
  _SOAStoreGeneric<float> mu;
  _SOAStoreGeneric<apbd::Shape> shape;
  _SOAStoreGeneric<float> density;
  _SOAStoreVec3 Mr;
  _SOAStoreGeneric<float> Mp;

  __host__ __device__ _SOAStoreBodyRigid() {}
  _SOAStoreBodyRigid(byte *data_store, size_t &offset, size_t count);
  static constexpr size_t size(size_t count) {
    return _SOAStoreVec7::size(count) * 5 + _SOAStoreVec3::size(count) * 2 +
           _SOAStoreQuaterion::size(count) * 2 +
           _SOAStoreGeneric<float>::size(count) * 3 +
           _SOAStoreGeneric<bool>::size(count) +
           _SOAStoreGeneric<apbd::Shape>::size(count);
  }

  __host__ __device__ void set(unsigned int index, const apbd::BodyRigid &data);
};

class SOAStore {
public:
  struct _SOAStoreBodyRigid BodyRigid;

  __host__ __device__ SOAStore() {}
  SOAStore(size_t body_rigid_count, size_t scene_count);

  void deallocate();
};

#ifdef __CUDA_ARCH__
#define global_store device_global_store
#else
#define global_store host_global_store
#endif
extern __device__ SOAStore device_global_store;
// dummy for host/device functions
extern SOAStore host_global_store;

} // namespace data
