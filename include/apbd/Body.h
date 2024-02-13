#pragma once
#include "apbd/Collisions.h"
#include "apbd/Shape.h"
#include <cstddef>
#include <cuda/std/array>

namespace apbd {
typedef Eigen::Matrix<float, 7, 1> vec7;
typedef Eigen::Matrix<float, 12, 1> vec12;

enum BODY_TYPE {
  BODY_AFFINE,
  BODY_RIGID,
};

struct BodyRigid {
  const static size_t DOF = 7;
  // base
  vec7 xInit;
  vec7 xdotInit;
  vec7 x;
  vec7 x0;
  vec7 x1;
  vec7 x1_0;
  vec7 dxJacobi;
  vec7 dxJacobiShock;
  bool collide;
  float mu;
  unsigned int layer;
  // rigid
  Shape shape;
  float density;
  Eigen::Vector3f Mr;
  float Mp;

  BodyRigid(Shape shape, float density);

  __host__ __device__ vec7 computeVelocity(unsigned int step,
                                           unsigned int substep, float hs);
  __host__ __device__ void computeInertiaConst();

  __host__ __device__ Eigen::Vector3f computePointVel(Eigen::Vector3f xl,
                                                      float hs);
  __host__ __device__ Eigen::Matrix4f computeTransform();
  __host__ __device__ void applyJacobi();
  __host__ __device__ void regularize();
};

struct BodyAffine {
  const static size_t DOF = 12;
  // base
  vec12 xInit;
  vec12 xdotInit;
  vec12 x;
  vec12 x0;
  vec12 x1;
  vec12 x1_0;
  vec12 dxJacobi;
  vec12 dxJacobiShock;
  bool collide;
  float mu;
  unsigned int layer;
  // rigid
  Shape shape;
  float density;
  Eigen::Vector3f Wa;
  float Wp;

  __host__ __device__ vec12 computeVelocity(unsigned int step,
                                            unsigned int substep, float hs);
  __host__ __device__ void computeInertiaConst();

  /*
   * Can only be called after calling setInitTransform
   */
  __host__ __device__ Eigen::Matrix4f computeInitTransform();
};

union _BodyInner {
  BodyAffine affine;
  BodyRigid rigid;
};

class Body {
public:
  BODY_TYPE type;
  _BodyInner data;

  Body(BodyRigid rigid);
  Body(BodyAffine affine);

  void init();

  __host__ __device__ bool collide();

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs, Eigen::Vector3f gravity);

  __host__ __device__ void clearShock();

  __host__ __device__ void applyJacobiShock();

  __host__ __device__ void regularize();

  __host__ __device__ void setInitTransform(Eigen::Matrix4f transform);

  __host__ __device__ void setInitVelocity(Eigen::Matrix<float, 6, 1> velocity);

  __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E);
  __host__
      __device__ cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>
      narrowphaseGround(Eigen::Matrix4f E);
  __host__ __device__ bool broadphaseRigid(Body *other);
  __host__
      __device__ cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
      narrowphaseRigid(Body *other);

  __host__ __device__ Eigen::Matrix4f computeTransform();
};

} // namespace apbd
