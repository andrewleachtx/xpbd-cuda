#include "data/soa.h"

namespace data {

SOAStore host_global_store;
__device__ SOAStore device_global_store;

SOAStore::SOAStore(size_t body_rigid_count) {
  // we need three aligned buffers of floats
  const size_t total_buffer_size = _SOAStoreBodyRigid::size(body_rigid_count);
  byte *const data_store = alloc_device<byte>(total_buffer_size);

  size_t offset = 0;
  this->BodyRigid = _SOAStoreBodyRigid(data_store, offset, body_rigid_count);
}

void SOAStore::deallocate() {}

_SOAStoreBodyRigid::_SOAStoreBodyRigid(byte *data_store, size_t &offset,
                                       size_t count)
    : xdotInit(data_store, offset, count) {}

void _SOAStoreBodyRigid::set(unsigned int index, const apbd::BodyRigid &data) {
  xdotInit.set(index, data.xInit);
}

_SOAStoreVec7::_SOAStoreVec7(byte *data_store, size_t &offset, size_t count) {
  this->x03 = get_aligned_buffer_segment<float4>(data_store, offset, count);
  this->x45 = get_aligned_buffer_segment<float2>(data_store, offset, count);
  this->x6 = get_aligned_buffer_segment<float>(data_store, offset, count);
}

vec7 _SOAStoreVec7::get(unsigned int index) {
  const float4 el03 = x03[index];
  const float2 el45 = x45[index];
  const float el6 = x6[index];
  return vec7(el03.x, el03.y, el03.z, el03.w, el45.x, el45.y, el6);
}

void _SOAStoreVec7::set(unsigned int index, vec7 new_val) {
  x03[index] = make_float4(new_val(0), new_val(1), new_val(2), new_val(3));
  x45[index] = make_float2(new_val(4), new_val(5));
  x6[index] = new_val(6);
}

} // namespace data
