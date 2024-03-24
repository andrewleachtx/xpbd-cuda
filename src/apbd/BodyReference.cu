#include "apbd/BodyReference.h"
#include "data/soa.h"
#include "se3/lib.h"

namespace apbd {

using Eigen::Vector3f, Eigen::Quaternionf;

#define IMPLEMENT_READONLY_ACCESS_FUNCTIONS(AttributeType, RefType, Type,      \
                                            attribute)                         \
  AttributeType RefType::attribute() const {                                   \
    return data::global_store.Type.attribute.get(index);                       \
  }
#define IMPLEMENT_ACCESS_FUNCTIONS(AttributeType, RefType, Type, attribute)    \
  AttributeType RefType::attribute() const {                                   \
    return data::global_store.Type.attribute.get(index);                       \
  }                                                                            \
  void RefType::attribute(AttributeType new_val) {                             \
    data::global_store.Type.attribute.set(index, new_val);                     \
  }

IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, xdotInit)
IMPLEMENT_ACCESS_FUNCTIONS(Vector3f, BodyRigidReference, BodyRigid, position)
IMPLEMENT_ACCESS_FUNCTIONS(Quaternionf, BodyRigidReference, BodyRigid, rotation)
vec7 BodyRigidReference::x() const {
  auto p = data::global_store.BodyRigid.position.get(index);
  auto q = data::global_store.BodyRigid.rotation.get(index).coeffs();
  return vec7(q(0), q(1), q(2), q(3), p(0), p(1), p(2));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x0)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, x1)
void BodyRigidReference::x1(Eigen::Vector4f new_q, Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.x1.set(index, vec7(new_q(0), new_q(1), new_q(2),
                                                  new_q(3), new_p(0), new_p(1),
                                                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(Quaternionf, BodyRigidReference, BodyRigid, x1_0_rot)
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobi)
void BodyRigidReference::dxJacobi(Eigen::Vector4f new_q,
                                  Eigen::Vector3f new_p) {
  data::global_store.BodyRigid.dxJacobi.set(
      index, vec7(new_q(0), new_q(1), new_q(2), new_q(3), new_p(0), new_p(1),
                  new_p(2)));
}
IMPLEMENT_ACCESS_FUNCTIONS(vec7, BodyRigidReference, BodyRigid, dxJacobiShock)
void BodyRigidReference::dxJacobiShock(Eigen::Vector4f new_q,
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

void BodyRigidReference::init(vec7 xInit) {
  this->computeInertiaConst();
  this->position(xInit.block<3, 1>(4, 0));
  this->rotation(Quaternionf(xInit.block<4, 1>(0, 0)));
  this->x0(xInit);
}

void BodyRigidReference::stepBDF1(unsigned int step, unsigned int substep,
                                  float hs, Eigen::Vector3f gravity) {
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
  t = t + se3::cross(Iw, w);                  // Coriolis
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

void BodyRigidReference::clearShock() {
  // TODO: sets layer to max
}

void BodyRigidReference::applyJacobiShock() {
  this->x1(this->x1() + this->dxJacobiShock());
  this->dxJacobiShock(vec7::Zero());
}

void BodyRigidReference::regularize() {
  auto x1_ = this->x1();
  this->position(x1_.block<3, 1>(4, 0));
  Eigen::Vector4f q = x1_.block<4, 1>(0, 0);
  q /= q.norm();
  this->rotation(Eigen::Quaternionf(q));
}

bool BodyRigidReference::broadphaseGround(Eigen::Matrix4f Eg) {
  Eigen::Matrix4f E = this->computeTransform();
  return this->shape().broadphaseGround(E, Eg);
}
GroundNarrowphaseReturn
BodyRigidReference::narrowphaseGround(Eigen::Matrix4f Eg) {
  Eigen::Matrix4f E = this->computeTransform();
  return this->shape().narrowphaseGround(E, Eg);
}
bool BodyRigidReference::broadphaseRigid(BodyRigidReference other) {
  Eigen::Matrix4f E1 = this->computeTransform();
  Eigen::Matrix4f E2 = other.computeTransform();
  return this->shape().broadphaseShape(E1, other.shape(), E2);
}
NarrowphaseReturn
BodyRigidReference::narrowphaseRigid(BodyRigidReference other) {
  Eigen::Matrix4f E1 = this->computeTransform();
  Eigen::Matrix4f E2 = other.computeTransform();
  return this->shape().narrowphaseShape(E1, other.shape(), E2);
}

Eigen::Matrix4f BodyRigidReference::computeTransform() {
  Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
  E.block<3, 3>(0, 0) = this->rotation().toRotationMatrix();
  E.block<3, 1>(0, 3) = this->position();
  return E;
}

vec7 BodyRigidReference::computeVelocity(unsigned int step,
                                         unsigned int substep, float hs) {
  if (step == 0 && substep == 0)
    return this->xdotInit();
  else
    return (this->x() - this->x0()) / hs;
}
void BodyRigidReference::computeInertiaConst() {
  auto d = this->density();
  auto s = this->shape();
  auto I = s.computeInertia(d);
  this->Mr(I.block<3, 1>(0, 0));
  this->Mp(I(4));
}

Eigen::Vector3f BodyRigidReference::computePointVel(Eigen::Vector3f xl,
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
void BodyRigidReference::applyJacobi() {
  this->x1(this->x1() + this->dxJacobi());
  this->dxJacobi(vec7::Zero());
  this->regularize();
}
void BodyRigidReference::write_state() {
  auto r = rotation().coeffs();
  printf("%f %f %f r %f %f %f %f", position()(0), position()(1), position()(2),
         r(0), r(1), r(2), r(3));
}

} // namespace apbd
