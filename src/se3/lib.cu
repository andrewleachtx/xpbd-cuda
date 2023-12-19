#include "se3/lib.h"
namespace se3 {

Eigen::Vector3f qdotToW(Eigen::Vector4f q, Eigen::Vector4f qdot) {
  // https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
  // Q = [
  // 	 q(3)  q(2) -q(1) -q(0)
  // 	-q(2)  q(3)  q(0) -q(1)
  // 	 q(1) -q(0)  q(3) -q(2)
  // 	];
  // w = 2*Q*qdot;
  return 2 *
         Eigen::Vector3f(
             q(3) * qdot(0) + q(2) * qdot(1) - q(1) * qdot(2) - q(0) * qdot(3),
             -q(2) * qdot(0) + q(3) * qdot(1) + q(0) * qdot(2) - q(1) * qdot(3),
             q(1) * qdot(0) - q(0) * qdot(1) + q(3) * qdot(2) - q(2) * qdot(3));
}

Eigen::Vector4f wToQdot(Eigen::Vector4f q, Eigen::Vector3f w) {
  // https://ahrs.readthedocs.io/en/latest/filters/angular.html#quaternion-derivative
  // W = [
  // 	    0  w(2) -w(1)  w(0)
  // 	-w(2)     0  w(0)  w(1)
  // 	 w(1) -w(0)     0  w(2)
  // 	-w(0) -w(1) -w(2)     0
  // 	];
  // qdot = 0.5*W*q;
  return 0.5 * Eigen::Vector4f(w(2) * q(1) - w(1) * q(2) + w(0) * q(3),
                               -w(2) * q(0) + w(0) * q(2) + w(1) * q(3),
                               w(1) * q(0) - w(0) * q(1) + w(2) * q(3),
                               -w(0) * q(0) - w(1) * q(1) - w(2) * q(2));
}
} // namespace se3
