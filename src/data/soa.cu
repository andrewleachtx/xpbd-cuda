#include "data/soa.h"
#include "util.h"

namespace data {

SOAStore global_store{};

using byte = unsigned char;

SOAStore::SOAStore() {}
SOAStore::SOAStore(size_t body_rigid_count) {
  // we need three aligned buffers of floats
  size_t count = get_aligned_size<float>(body_rigid_count) * 3;
  byte *data_store = alloc_device<byte>(count);

  this->body_rigid.x0 = reinterpret_cast<float *>(data_store);

  size_t offset = get_aligned_size<float>(body_rigid_count);
  this->body_rigid.x1 = reinterpret_cast<float *>(data_store + offset);

  offset += get_aligned_size<float>(body_rigid_count);
  this->body_rigid.x2 = reinterpret_cast<float *>(data_store + offset);
}

void SOAStore::deallocate() {}

} // namespace data
