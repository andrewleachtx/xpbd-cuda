#pragma once
#include "Body.h"
#include "Constraint.h"
#include "Model.h"

namespace apbd {

class Model;
class Collider {
  size_t bp_cap_1;
  size_t bp_cap_2;
  size_t bp_count_1;
  size_t bp_count_2;
  Body **bpList1;
  Body **bpList2;

public:
  size_t collision_count;
  Constraint *collisions;

  __device__ __host__ void broadphase(Model *model);
  __device__ __host__ void narrowphase(Model *model);

  Collider(Model *model);
  __device__ __host__ Collider(Model *model, size_t scene_id,
                               Body **body_ptr_buffer,
                               Constraint *constraint_buffer);
  __device__ __host__ void run(Model *model);
  static void allocate_buffers(Model &model, int sim_count,
                               Body **&body_ptr_buffer,
                               Constraint *&constraint_buffer);
  static __device__ __host__ std::pair<Eigen::Vector3f, Eigen::Vector3f>
  generateTangents(Eigen::Vector3f nor);
};
} // namespace apbd
