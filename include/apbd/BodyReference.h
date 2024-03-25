#pragma once
#include "apbd/Body.h"
#include "data/soa.h"
#include "se3/lib.h"
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

using Eigen::Vector3f, Eigen::Quaternionf;

#define IMPLEMENT_READONLY_ACCESS_FUNCTIONS(AttributeType, RefType, Type,      \
                                            attribute)                         \
  inline AttributeType RefType::attribute() const {                            \
    return data::global_store.Type.attribute.get(index);                       \
  }
#define IMPLEMENT_ACCESS_FUNCTIONS(AttributeType, RefType, Type, attribute)    \
  inline AttributeType RefType::attribute() const {                            \
    return data::global_store.Type.attribute.get(index);                       \
  }                                                                            \
  inline void RefType::attribute(AttributeType new_val) {                      \
    data::global_store.Type.attribute.set(index, new_val);                     \
  }

IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, xdotInit)
IMPLEMENT_ACCESS_FUNCTIONS(Vector3f, BodyRigidReference, BodyRigid, position)
IMPLEMENT_ACCESS_FUNCTIONS(Quaternionf, BodyRigidReference, BodyRigid, rotation)
inline vec7 BodyRigidReference::x() const {
  auto p = data::global_store.BodyRigid.position.get(index);
  auto q = data::global_store.BodyRigid.rotation.get(index).coeffs();
  return vec7(q(0), q(1), q(2), q(3), p(0), p(1), p(2));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x0)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x1)
inline void BodyRigidReference::x1(Eigen::Vector4f new_q,
                                   Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.x1.set(index, vec7(new_q(0), new_q(1), new_q(2),
                                                  new_q(3), new_p(0), new_p(1),
                                                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(Quaternionf, BodyRigidReference, BodyRigid, x1_0_rot)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobi)
inline void BodyRigidReference::dxJacobi(Eigen::Vector4f new_q,
                                         Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.dxJacobi.set(
      index, vec7(new_q(0), new_q(1), new_q(2), new_q(3), new_p(0), new_p(1),
                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobiShock)
inline void BodyRigidReference::dxJacobiShock(Eigen::Vector4f new_q,
                                              Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.dxJacobiShock.set(
      index, vec7(new_q(0), new_q(1), new_q(2), new_q(3), new_p(0), new_p(1),
                  new_p(2)));
}
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(bool, BodyRigidReference, BodyRigid,
                                    collide)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, mu)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(Shape, BodyRigidReference, BodyRigid, shape)
IMPLEMENT_READONLY_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid,
                                    density)
IMPLEMENT_ACCESS_FUNCTIONS(Vector3f, BodyRigidReference, BodyRigid, Mr)
IMPLEMENT_ACCESS_FUNCTIONS(float, BodyRigidReference, BodyRigid, Mp)

inline void BodyRigidReference::init(vec7 xInit) {
  this->computeInertiaConst();
  this->position(xInit.block<3, 1>(4, 0));
  this->rotation(Quaternionf(xInit.block<4, 1>(0, 0)));
  this->x0(xInit);
}

inline void BodyRigidReference::stepBDF1(unsigned int step,
                                         unsigned int substep, float hs,
                                         Eigen::Vector3f gravity) {
  auto xdot = this->computeVelocity(step, substep, hs);
  Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
  Eigen::Vector3f v = xdot.block<3, 1>(4, 0); // pdot
  this->x0(this->x());
  Eigen::Vector4f q = this->rotation().coeffs();
  Eigen::Vector3f p = this->position();
  auto w = se3::qdotToW(q, qdot); // angular velocity in body coords
  Vector3f f = Vector3f::Zero();  // translational force in world space
  Vector3f t = Vector3f::Zero();  // angular torque in body space
  auto m = this->Mp();            // scalar mass
  auto I = this->Mr();            // inertia in body space
  Eigen::Vector3f Iw = I.array() * w.array(); // angular momentum in body space
  f = f + m * gravity;                        // Gravity
  t = t + Iw.cross(w);                        // Coriolis
  // Integrate velocities
  w = w + hs * Eigen::Vector3f(t.array() / I.array());
  v = v + hs * (f / m);
  qdot = se3::wToQdot(q, w);
  // Integrate positions
  q = q + hs * qdot;
  p = p + hs * v;
  q = q / q.norm();
  this->rotation(Eigen::Quaternionf(q));
  this->position(p);
  this->x1_0_rot(this->rotation());
  this->x1(this->rotation().coeffs(), this->position());
}

inline void BodyRigidReference::clearShock() {
  // TODO: sets layer to max
}

inline void BodyRigidReference::applyJacobiShock() {
  this->x1(this->x1() + this->dxJacobiShock());
  this->dxJacobiShock(vec7::Zero());
}

inline void BodyRigidReference::regularize() {
  auto x1_ = this->x1();
  this->position(x1_.block<3, 1>(4, 0));
  Eigen::Vector4f q = x1_.block<4, 1>(0, 0);
  q /= q.norm();
  this->rotation(Eigen::Quaternionf(q));
}

inline bool BodyRigidReference::broadphaseGround(Eigen::Matrix4f Eg) {
  Eigen::Matrix4f E = this->computeTransform();
  return this->shape().broadphaseGround(E, Eg);
}
inline GroundNarrowphaseReturn
BodyRigidReference::narrowphaseGround(Eigen::Matrix4f Eg) {
  Eigen::Matrix4f E = this->computeTransform();
  return this->shape().narrowphaseGround(E, Eg);
}
inline bool BodyRigidReference::broadphaseRigid(BodyRigidReference other) {
  Eigen::Matrix4f E1 = this->computeTransform();
  Eigen::Matrix4f E2 = other.computeTransform();
  return this->shape().broadphaseShape(E1, other.shape(), E2);
}
inline NarrowphaseReturn
BodyRigidReference::narrowphaseRigid(BodyRigidReference other) {
  Eigen::Matrix4f E1 = this->computeTransform();
  Eigen::Matrix4f E2 = other.computeTransform();
  return this->shape().narrowphaseShape(E1, other.shape(), E2);
}

inline Eigen::Matrix4f BodyRigidReference::computeTransform() {
  Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
  E.block<3, 3>(0, 0) = this->rotation().toRotationMatrix();
  E.block<3, 1>(0, 3) = this->position();
  return E;
}

inline vec7 BodyRigidReference::computeVelocity(unsigned int step,
                                                unsigned int substep,
                                                float hs) {
  if (step == 0 && substep == 0)
    return this->xdotInit();
  else
    return (this->x() - this->x0()) / hs;
}
inline void BodyRigidReference::computeInertiaConst() {
  auto d = this->density();
  auto s = this->shape();
  auto I = s.computeInertia(d);
  this->Mr(I.block<3, 1>(0, 0));
  this->Mp(I(4));
}

inline Eigen::Vector3f BodyRigidReference::computePointVel(Eigen::Vector3f xl,
                                                           float hs) {
  // xdot = this.computeVelocity(k,ks,hs);
  vec7 xdot = (this->x() - this->x0()) / hs;
  Eigen::Vector4f qdot = xdot.block<4, 1>(0, 0);
  Eigen::Vector3f pdot = xdot.block<3, 1>(4, 0); // in world coords
  Eigen::Quaternionf q = this->rotation();
  Eigen::Vector3f w =
      se3::qdotToW(q.coeffs(), qdot); // angular velocity in body coords
  // v = R*cross(w,xl) + pdot
  return (q * w.cross(xl)) + pdot;
}
inline void BodyRigidReference::applyJacobi() {
  this->x1(this->x1() + this->dxJacobi());
  this->dxJacobi(vec7::Zero());
  this->regularize();
}
inline void BodyRigidReference::write_state() {
  auto r = rotation().coeffs();
  printf("%f %f %f r %f %f %f %f", position()(0), position()(1), position()(2),
         r(0), r(1), r(2), r(3));
}

} // namespace apbd
