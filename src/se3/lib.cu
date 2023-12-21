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

Eigen::Matrix3f aaToMat(Eigen::Vector3f axis, float angle) {
  // Create a rotation matrix from an (axis,angle) pair
  // From vecmath
  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  auto ax = axis(1);
  auto ay = axis(2);
  auto az = axis(3);
  auto mag = sqrt(ax * ax + ay * ay + az * az);
  if (mag > se3::THRESH) {
    mag = 1.0 / mag;
    ax = ax * mag;
    ay = ay * mag;
    az = az * mag;
    if (abs(ax) < se3::THRESH && abs(ay) < se3::THRESH) {
      // Rotation about Z
      if (az < 0) {
        angle = -angle;
      }
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      R(1, 1) = cosTheta;
      R(1, 2) = -sinTheta;
      R(2, 1) = sinTheta;
      R(2, 2) = cosTheta;
    } else if (abs(ay) < se3::THRESH && abs(az) < se3::THRESH) {
      // Rotation about X
      if (ax < 0) {
        angle = -angle;
      }
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      R(2, 2) = cosTheta;
      R(2, 3) = -sinTheta;
      R(3, 2) = sinTheta;
      R(3, 3) = cosTheta;
    } else if (abs(az) < se3::THRESH && abs(ax) < se3::THRESH) {
      // Rotation about Y
      if (ay < 0) {
        angle = -angle;
      }
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      R(1, 1) = cosTheta;
      R(1, 3) = sinTheta;
      R(3, 1) = -sinTheta;
      R(3, 3) = cosTheta;
    } else {
      // General rotation
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      auto t = 1.0 - cosTheta;
      auto xz = ax * az;
      auto xy = ax * ay;
      auto yz = ay * az;
      R(1, 1) = t * ax * ax + cosTheta;
      R(1, 2) = t * xy - sinTheta * az;
      R(1, 3) = t * xz + sinTheta * ay;
      R(2, 1) = t * xy + sinTheta * az;
      R(2, 2) = t * ay * ay + cosTheta;
      R(2, 3) = t * yz - sinTheta * ax;
      R(3, 1) = t * xz - sinTheta * ay;
      R(3, 2) = t * yz + sinTheta * ax;
      R(3, 3) = t * az * az + cosTheta;
    }
  }
  return R;
}

} // namespace se3
