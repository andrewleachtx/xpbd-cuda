#pragma once
#include "apbd/Body.h"
#include "util.h"

namespace apbd {

using GroundNarrowphaseReturn =
    cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>;
using NarrowphaseReturn =
    cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>;

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

  __host__ __device__ vec7 x() const;
  __host__ __device__ vec7 xdotInit() const;
  __host__ __device__ void xdotInit(vec7 new_val);
  __host__ __device__ Eigen::Vector3f position() const;
  __host__ __device__ void position(Eigen::Vector3f new_val);
  __host__ __device__ Eigen::Quaternionf rotation() const;
  __host__ __device__ void rotation(Eigen::Quaternionf new_val);
  __host__ __device__ vec7 x0() const;
  __host__ __device__ void x0(vec7 new_val);
  __host__ __device__ vec7 x1() const;
  __host__ __device__ void x1(vec7 new_val);
  __host__ __device__ void x1(Eigen::Vector4f new_q, Eigen::Vector3f new_p);
  __host__ __device__ Eigen::Quaternionf x1_0_rot() const;
  __host__ __device__ void x1_0_rot(Eigen::Vector4f new_val);
  __host__ __device__ void x1_0_rot(Eigen::Quaternionf new_val);
  __host__ __device__ vec7 dxJacobi() const;
  __host__ __device__ void dxJacobi(vec7 new_val);
  __host__ __device__ void dxJacobi(Eigen::Vector4f new_q,
                                    Eigen::Vector3f new_p);
  __host__ __device__ vec7 dxJacobiShock() const;
  __host__ __device__ void dxJacobiShock(vec7 new_val);
  __host__ __device__ void dxJacobiShock(Eigen::Vector4f new_q,
                                         Eigen::Vector3f new_p);
  __host__ __device__ Eigen::Vector3f Mr() const;
  __host__ __device__ void Mr(Eigen::Vector3f new_val);
  __host__ __device__ float Mp() const;
  __host__ __device__ void Mp(float new_val);
  // readonly elements
  __host__ __device__ bool collide() const;
  __host__ __device__ float mu() const;
  // __host__ __device__ unsigned int layer();
  __host__ __device__ Shape shape() const;
  __host__ __device__ float density() const;

  // delegated implementations
  __host__ __device__ void init(vec7 xInit);

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs, Eigen::Vector3f gravity);

  __host__ __device__ void clearShock();

  __host__ __device__ void applyJacobiShock();

  __host__ __device__ void regularize();

  __host__ __device__ void setInitTransform(Eigen::Matrix4f transform);

  __host__ __device__ void setInitVelocity(Eigen::Matrix<float, 6, 1> velocity);

  __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E);
  __host__ __device__ GroundNarrowphaseReturn
  narrowphaseGround(Eigen::Matrix4f E);
  __host__ __device__ bool broadphaseRigid(BodyRigidReference other);
  __host__ __device__ NarrowphaseReturn
  narrowphaseRigid(BodyRigidReference other);

  __host__ __device__ Eigen::Matrix4f computeTransform();

  __host__ __device__ vec7 computeVelocity(unsigned int step,
                                           unsigned int substep, float hs);
  __host__ __device__ void computeInertiaConst();

  __host__ __device__ Eigen::Vector3f computePointVel(Eigen::Vector3f xl,
                                                      float hs);
  __host__ __device__ void applyJacobi();
  __host__ __device__ void write_state();
};

class BodyAffineReference { /* TODO */
};

class BodyReference {
public:
  // using 32 bit because it is large enough
  unsigned int index;
  BODY_TYPE type;

  __host__ __device__ BodyReference() {}
  __host__ __device__ BodyReference(unsigned int index, BODY_TYPE type)
      : index(index), type(type) {}
  __host__ __device__ BodyRigidReference get_rigid() const {
    return BodyRigidReference(index);
  }
  __host__ __device__ BodyAffineReference get_affine() const {
    return BodyAffineReference(/*TODO*/);
  }

  // void init(const Body &body) {
  //   switch (type) {
  //   case BODY_RIGID: {
  //     auto data = get_rigid();
  //     data.init(body.data.rigid.xInit);
  //   }
  //   case BODY_AFFINE: {
  //     auto data = get_affine(); /* TODO */
  //   }
  //   default: {
  //   }
  //   }
  // }
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ bool collide() const,
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
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ GroundNarrowphaseReturn
                                   narrowphaseGround(Eigen::Matrix4f E),
                               data.narrowphaseGround(E));
  // TODO: handle other body types
  IMPLEMENT_DELEGATED_FUNCTION(
      __host__ __device__ bool broadphaseRigid(BodyReference other),
      data.broadphaseRigid(other.get_rigid()));
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ NarrowphaseReturn
                                   narrowphaseRigid(BodyReference other),
                               data.narrowphaseRigid(other.get_rigid()));

  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__
                                   Eigen::Matrix4f computeTransform(),
                               data.computeTransform());
  IMPLEMENT_DELEGATED_FUNCTION(__host__ __device__ void write_state(),
                               data.write_state());
};

} // namespace apbd
