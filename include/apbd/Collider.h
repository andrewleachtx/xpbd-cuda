#pragma once
#include "Model.h"
#include "Body.h"
#include "Constraint.h"

class Model;

namespace apbd {
class Collider {
  size_t bp_cap_1;
  size_t bp_cap_2;
  size_t bp_count_1;
  size_t bp_count_2;
  Body **bpList1;
  Body **bpList2;
  size_t collision_count;
  Constraint *collisions;

  __device__ __host__ void broadphase();
  __device__ __host__ void narrowphase();

public:
  Collider(Model *model);
  __device__ __host__ void run(Model* model);
  static __device__ __host__ std::pair<Eigen::Vector3f, Eigen::Vector3f>
  generateTangents(Eigen::Vector3f nor);
};
} // namespace apbd
