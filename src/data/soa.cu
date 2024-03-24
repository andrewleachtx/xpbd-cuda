#include "data/soa.h"

namespace data {

size_t soa_index(unsigned int index) {
#ifdef __CUDA_ARCH__
  const size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t scene_count = blockDim.x * gridDim.x;
  return scene_id + index * scene_count;
#else
  return _thread_scene_id + index * _global_scene_count;
#endif
}

SOAStore host_global_store;
__device__ SOAStore device_global_store;

SOAStore::SOAStore(size_t body_rigid_count, size_t scene_count) {
  const size_t alignment_count = BLOCK_SIZE;
  size_t aligned_scene_count = 0;
  if (scene_count % alignment_count == 0)
    aligned_scene_count = scene_count;
  else
    aligned_scene_count = (scene_count / alignment_count + 1) * alignment_count;

  const size_t object_count = body_rigid_count * aligned_scene_count;
  // we need three aligned buffers of floats
  const size_t total_buffer_size = _SOAStoreBodyRigid::size(object_count);
  byte *const data_store = alloc_device<byte>(total_buffer_size);

  size_t offset = 0;
  this->BodyRigid = _SOAStoreBodyRigid(data_store, offset, object_count);
}

void SOAStore::deallocate() {}

_SOAStoreBodyRigid::_SOAStoreBodyRigid(byte *data_store, size_t &offset,
                                       size_t count)
    : xdotInit(data_store, offset, count), position(data_store, offset, count),
      rotation(data_store, offset, count), x0(data_store, offset, count),
      x1(data_store, offset, count), x1_0_rot(data_store, offset, count),
      dxJacobi(data_store, offset, count),
      dxJacobiShock(data_store, offset, count),
      collide(data_store, offset, count), mu(data_store, offset, count),
      shape(data_store, offset, count), density(data_store, offset, count),
      Mr(data_store, offset, count), Mp(data_store, offset, count) {}

void _SOAStoreBodyRigid::set(unsigned int index, const apbd::BodyRigid &data) {
  xdotInit.set(index, data.xInit);
  position.set(index, data.x.block<3, 1>(4, 0));
  rotation.set(index, Eigen::Quaternionf(data.x.block<4, 1>(0, 0)));
  x0.set(index, data.x0);
  x1.set(index, data.x1);
  x1_0_rot.set(index, Eigen::Quaternionf(data.x1_0.block<4, 1>(0, 0)));
  dxJacobi.set(index, data.dxJacobi);
  dxJacobiShock.set(index, data.dxJacobiShock);
  collide.set(index, data.collide);
  mu.set(index, data.mu);
  shape.set(index, data.shape);
  density.set(index, data.density);
  Mr.set(index, data.Mr);
  Mp.set(index, data.Mp);
}

_SOAStoreQuaterion::_SOAStoreQuaterion(byte *data_store, size_t &offset,
                                       size_t count) {
  this->data = get_aligned_buffer_segment<float4>(data_store, offset, count);
}

Eigen::Quaternionf _SOAStoreQuaterion::get(unsigned int index) const {
  const float4 data_val = data[soa_index(index)];
  return Eigen::Quaternionf(data_val.x, data_val.y, data_val.z, data_val.w);
}

void _SOAStoreQuaterion::set(unsigned int index, Eigen::Quaternionf new_val) {
  const auto coeffs = new_val.coeffs();
  data[soa_index(index)] =
      make_float4(coeffs(3), coeffs(0), coeffs(1), coeffs(2));
}

_SOAStoreVec3::_SOAStoreVec3(byte *data_store, size_t &offset, size_t count) {
  this->x01 = get_aligned_buffer_segment<float2>(data_store, offset, count);
  this->x2 = get_aligned_buffer_segment<float>(data_store, offset, count);
}

Eigen::Vector3f _SOAStoreVec3::get(unsigned int index) const {
  const float2 el01 = x01[soa_index(index)];
  const float el2 = x2[soa_index(index)];
  return Eigen::Vector3f(el01.x, el01.y, el2);
}

void _SOAStoreVec3::set(unsigned int index, Eigen::Vector3f new_val) {
  x01[soa_index(index)] = make_float2(new_val(0), new_val(1));
  x2[soa_index(index)] = new_val(2);
}

_SOAStoreVec7::_SOAStoreVec7(byte *data_store, size_t &offset, size_t count) {
  this->x03 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x45 = get_aligned_buffer_segment<float2>(data_store, offset, count);
  this->x6 = get_aligned_buffer_segment<float>(data_store, offset, count);
}

vec7 _SOAStoreVec7::get(unsigned int index) const {
  const float4 el03 = x03[soa_index(index)];
  const float2 el45 = x45[soa_index(index)];
  const float el6 = x6[soa_index(index)];
  return vec7(el03.x, el03.y, el03.z, el03.w, el45.x, el45.y, el6);
}

void _SOAStoreVec7::set(unsigned int index, vec7 new_val) {
  x03[soa_index(index)] =
      make_float4(new_val(0), new_val(1), new_val(2), new_val(3));
  x45[soa_index(index)] = make_float2(new_val(4), new_val(5));
  x6[soa_index(index)] = new_val(6);
}

} // namespace data
