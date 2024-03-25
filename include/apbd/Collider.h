#pragma once
#include "BodyReference.h"
#include "Constraint.h"
#include "Model.h"

namespace apbd {

class Model;
class Collider {
  size_t bp_cap_1;
  size_t bp_cap_2;
  size_t bp_count_1;
  size_t bp_count_2;
  BodyReference *bpList1;
  BodyReference *bpList2;

public:
  size_t collision_count;
  Constraint *collisions;

  __device__ __host__ void broadphase(Model *model);
  __device__ __host__ void narrowphase(Model *model);

  Collider(Model *model);
  __device__ __host__ Collider(Model *model, size_t scene_id,
                               BodyReference *body_ptr_buffer,
                               Constraint *constraint_buffer);
  __device__ __host__ void run(Model *model);
  static void allocate_buffers(Model &model, int sim_count,
                               BodyReference *&body_ptr_buffer,
                               Constraint *&constraint_buffer);
  static __device__ __host__ void generateTangents(const Eigen::Vector3f nor,
                                                   Eigen::Vector3f *out_tx,
                                                   Eigen::Vector3f *out_ty);
};

inline void Collider::generateTangents(const Eigen::Vector3f nor,
                                       Eigen::Vector3f *tanx,
                                       Eigen::Vector3f *tany) {
  Eigen::Vector3f tmp;
  if (abs(nor(2)) < 1e-6) {
    tmp << 0, 0, 1;
  } else {
    tmp << 1, 0, 0;
  }
  *tany = nor.cross(tmp);
  *tany = *tany / tany->norm();
  *tanx = tany->cross(nor);
}

} // namespace apbd
