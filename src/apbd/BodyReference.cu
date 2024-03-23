#include "apbd/BodyReference.h"
#include "data/soa.h"

namespace apbd {

vec7 BodyRigidReference::xdotInit() {
  return data::global_store.BodyRigid.xdotInit.get(index);
}

void BodyRigidReference::xdotInit(vec7 new_val) {
  data::global_store.BodyRigid.xdotInit.set(index, new_val);
}

} // namespace apbd
