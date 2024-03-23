#pragma once
#include "apbd/Body.h"

namespace apbd {

using NarrowphaseReturn =
    cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>;

#define IMPLEMENT_DELEGATED_FUNCTION(signature, call)                          \
  signature {                                                                  \
    switch (type) {                                                            \
    case BODY_RIGID: {                                                         \
      auto data = get_rigid();                                                 \
      return call;                                                             \
    }                                                                          \
    case BODY_AFFINE: {                                                        \
      auto data = get_affine(); /* TODO: return call; */                       \
    }                                                                          \
    default: {                                                                 \
    }                                                                          \
    }                                                                          \
  }

class BodyRigidReference {
  unsigned int index;

public:
  __host__ __device__ BodyRigidReference(unsigned int index) : index(index) {}
  // access the data elements in BodyRigid

  __host__ __device__ vec7 xdotInit();
  __host__ __device__ void xdotInit(vec7 new_val);
  __host__ __device__ Eigen::Vector3f position();
  __host__ __device__ void position(Eigen::Vector3f new_val);
  __host__ __device__ Eigen::Quaternionf rotation();
  __host__ __device__ void rotation(Eigen::Quaternionf new_val);
  __host__ __device__ vec7 x0();
  __host__ __device__ void x0(vec7 new_val);
  __host__ __device__ vec7 x1();
  __host__ __device__ void x1(vec7 new_val);
  __host__ __device__ vec7 x1_0();
  __host__ __device__ void x1_0(vec7 new_val);
  __host__ __device__ vec7 dxJacobi();
  __host__ __device__ void dxJacobi(vec7 new_val);
  __host__ __device__ vec7 dxJacobiShock();
  __host__ __device__ void dxJacobiShock(vec7 new_val);
  // readonly elements
  __host__ __device__ bool collide();
  __host__ __device__ float mu();
  // __host__ __device__ unsigned int layer();
  __host__ __device__ Shape shape();
  __host__ __device__ float density();
  __host__ __device__ Eigen::Vector3f Mr();
  __host__ __device__ Eigen::Vector3f Mp();

  // delegated implementations
  void init();

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs, Eigen::Vector3f gravity);

  __host__ __device__ void clearShock();

  __host__ __device__ void applyJacobiShock();

  __host__ __device__ void regularize();

  __host__ __device__ void setInitTransform(Eigen::Matrix4f transform);

  __host__ __device__ void setInitVelocity(Eigen::Matrix<float, 6, 1> velocity);

  __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E);
  __host__ __device__ NarrowphaseReturn narrowphaseGround(Eigen::Matrix4f E);
  __host__ __device__ bool broadphaseRigid(Body *other);
  __host__ __device__ NarrowphaseReturn narrowphaseRigid(Body *other);

  __host__ __device__ Eigen::Matrix4f computeTransform();

  __host__ __device__ void write_state();
};

class BodyAffineReference { /* TODO */
};

class BodyReference {
  // using 32 bit because it is large enough
  unsigned int index;
  BODY_TYPE type;

public:
  __host__ __device__ BodyReference(unsigned int index, BODY_TYPE type)
      : index(index), type(type) {}
  __host__ __device__ BodyRigidReference get_rigid() {
    return BodyRigidReference(index);
  }
  __host__ __device__ BodyAffineReference get_affine() {
    return BodyAffineReference(/*TODO*/);
  }

  IMPLEMENT_DELEGATED_FUNCTION(void init(), data.init());
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ bool collide(),
                               data.collide());
  IMPLEMENT_DELEGATED_FUNCTION(
      __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                        float hs, Eigen::Vector3f gravity),
      data.stepBDF1(step, substep, hs, gravity));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void clearShock(),
                               data.clearShock());
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void applyJacobiShock(),
                               data.applyJacobiShock());
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void regularize(),
                               data.regularize());
  IMPLEMENT_DELEGATED_FUNCTION(
      __host__ __device__ void setInitTransform(Eigen::Matrix4f transform),
      data.setInitTransform(transform));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void setInitVelocity(
                                   Eigen::Matrix<float, 6, 1> velocity),
                               data.setInitVelocity(velocity));
  IMPLEMENT_DELEGATED_FUNCTION(
      __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E),
      data.broadphaseGround(E));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ NarrowphaseReturn
                                   narrowphaseGround(Eigen::Matrix4f E),
                               data.narrowphaseGround(E));
  IMPLEMENT_DELEGATED_FUNCTION(
      __host__ __device__ bool broadphaseRigid(Body *other),
      data.broadphaseRigid(other));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ NarrowphaseReturn
                                   narrowphaseRigid(Body *other),
                               data.narrowphaseRigid(other));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__
                                   Eigen::Matrix4f computeTransform(),
                               data.computeTransform());
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void write_state(),
                               data.write_state());
};

} // namespace apbd
