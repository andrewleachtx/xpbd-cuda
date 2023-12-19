#include "apbd/Body.h"
#include <Eigen/SVD>
#include <limits>

namespace apbd {

using Eigen::seq, Eigen::Vector3f;

Body::Body(BodyRigid rigid) : type(BODY_RIGID), data{.rigid{rigid}} {}
Body::Body(BodyAffine affine)
    : type(BODY_AFFINE), data{.affine = std::move(affine)} {}

void Body::stepBDF1(unsigned int step, unsigned int substep, float hs,
                    Eigen::Vector3f gravity) {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    auto xdot = data.computeVelocity(step, substep, hs);
    data.x0 = data.x;
    auto axdot = xdot(seq(0, 2));
    auto aydot = xdot(seq(3, 5));
    auto azdot = xdot(seq(6, 8));
    auto pdot = xdot(seq(9, 11));
    auto w = data.Wp;
    auto W = data.Wa;
    auto f = Eigen::Vector3f(0);
    auto t = Eigen::Matrix<float, 9, 1>(0); // affine torque
    auto tx = t(seq(0, 2));
    auto ty = t(seq(3, 5));
    auto tz = t(seq(6, 8));
    f = f + gravity / w; // Gravity
    // Integrate velocities
    axdot = axdot + hs * Eigen::Vector3f(W.array() * tx.array());
    aydot = aydot + hs * Eigen::Vector3f(W.array() * ty.array());
    azdot = azdot + hs * Eigen::Vector3f(W.array() * tz.array());
    pdot = pdot + hs * (w * f);
    // Integrate positions
    data.x(seq(0, 2)) = data.x(seq(0, 2)) + hs * axdot;
    data.x(seq(3, 5)) = data.x(seq(3, 5)) + hs * aydot;
    data.x(seq(6, 8)) = data.x(seq(6, 8)) + hs * azdot;
    data.x(seq(9, 11)) = data.x(seq(9, 11)) + hs * pdot;

    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    auto xdot = data.computeVelocity(step, substep, hs);
    auto qdot = xdot(seq(0, 3));
    auto v = xdot(seq(4, 6)); // pdot
    data.x0 = data.x;
    auto q = data.x(seq(0, 3));
    auto p = data.x(seq(4, 6));
    auto w = se3::qdotToW(q, qdot); // angular velocity in body coords
    auto f = Vector3f(0);           // translational force in world space
    auto t = Vector3f(0);           // angular torque in body space
    auto m = data.Mp;               // scalar mass
    auto I = data.Mr;               // inertia in body space
    Eigen::Vector3f Iw =
        I.array() * w.array(); // angular momentum in body space
    f = f + m * gravity;       // Gravity
    t = t + Iw.cross(w);       // Coriolis
    // Integrate velocities
    w = w + hs * Eigen::Vector3f(t.array() / I.array());
    v = v + hs * (f / m);
    qdot = se3::wToQdot(q, w);
    // Integrate positions
    q = q + hs * qdot;
    p = p + hs * v;
    q = q / q.norm();
    data.x(seq(0, 3)) = q;
    data.x(seq(4, 6)) = p;
    data.x1_0 = data.x;
    data.x1 = data.x1_0;
    break;
  }
  default:
    break;
  }
}

void Body::clearShock() {
  // set layer to MAX
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.layer = std::numeric_limits<unsigned int>::max();
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.layer = std::numeric_limits<unsigned int>::max();
    break;
  }
  default:
    break;
  }
}

void Body::applyJacobiShock() {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.x1 += data.dxJacobiShock;
    data.dxJacobiShock = vec12(0);
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.x1 += data.dxJacobiShock;
    data.dxJacobiShock = vec7(0);
    break;
  }
  default:
    break;
  }
  this->regularize();
}

void Body::regularize() {
  switch (this->type) {
  case BODY_AFFINE: {
    auto &data = this->data.affine;
    data.x = data.x1;
    Eigen::Matrix3f A = data.x.reshaped(3, 3);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(A);
    A = svd.matrixU() * svd.matrixV().transpose();
    data.x = A.reshaped(9, 1);
    break;
  }
  case BODY_RIGID: {
    auto &data = this->data.rigid;
    data.x = data.x1;
    Eigen::Vector4f q = data.x(seq(0, 4));
    q /= q.norm();
    data.x(seq(0, 4)) = q;
    break;
  }
  default:
    break;
  }
}

BodyRigid::BodyRigid(Shape shape, float density)
    : shape(shape), density(density) {}

vec7 BodyRigid::computeVelocity(unsigned int step, unsigned int substep,
                                float hs) {
  if (step == 0 && substep == 0)
    return this->xDotInit;
  else
    return (this->x - this->x0) / hs;
}
vec12 BodyAffine::computeVelocity(unsigned int step, unsigned int substep,
                                  float hs) {
  if (step == 0 && substep == 0)
    return this->xDotInit;
  else
    return (this->x - this->x0) / hs;
}
} // namespace apbd
