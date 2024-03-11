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
  auto ax = axis(0);
  auto ay = axis(1);
  auto az = axis(2);
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
      R(0, 0) = cosTheta;
      R(0, 1) = -sinTheta;
      R(1, 0) = sinTheta;
      R(1, 1) = cosTheta;
    } else if (abs(ay) < se3::THRESH && abs(az) < se3::THRESH) {
      // Rotation about X
      if (ax < 0) {
        angle = -angle;
      }
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      R(1, 1) = cosTheta;
      R(1, 2) = -sinTheta;
      R(2, 1) = sinTheta;
      R(2, 2) = cosTheta;
    } else if (abs(az) < se3::THRESH && abs(ax) < se3::THRESH) {
      // Rotation about Y
      if (ay < 0) {
        angle = -angle;
      }
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      R(0, 0) = cosTheta;
      R(0, 2) = sinTheta;
      R(2, 0) = -sinTheta;
      R(2, 2) = cosTheta;
    } else {
      // General rotation
      auto sinTheta = sin(angle);
      auto cosTheta = cos(angle);
      auto t = 1.0 - cosTheta;
      auto xz = ax * az;
      auto xy = ax * ay;
      auto yz = ay * az;
      R(0, 0) = t * ax * ax + cosTheta;
      R(0, 1) = t * xy - sinTheta * az;
      R(0, 2) = t * xz + sinTheta * ay;
      R(1, 0) = t * xy + sinTheta * az;
      R(1, 1) = t * ay * ay + cosTheta;
      R(1, 2) = t * yz - sinTheta * ax;
      R(2, 0) = t * xz - sinTheta * ay;
      R(2, 1) = t * yz + sinTheta * ax;
      R(2, 2) = t * az * az + cosTheta;
    }
  }
  return R;
}

// Eigen::Vector4f matToQ(Eigen::Matrix3f R) {
//   double m00, m01, m02, m10, m11, m12, m20, m21, m22;
//
//   m00 = R(1,1); m01 = R(1,2); m02 = R(1,3);
//   m10 = R(2,1); m11 = R(2,2); m12 = R(2,3);
//   m20 = R(3,1); m21 = R(3,2); m22 = R(3,3);
//   double tr = m00 + m11 + m22;
//   double qw, qx, qy, qz;
//   if (tr > 0) {
//     double S = sqrt(tr+1.0) * 2; // S=4*qw
//     qw = 0.25 * S;
//     qx = (m21 - m12) / S;
//     qy = (m02 - m20) / S;
//     qz = (m10 - m01) / S;
//   }else if ((m00 > m11)&&(m00 > m22)){
//     double S = sqrt(1.0 + m00 - m11 - m22) * 2; // S=4*qx
//     qw = (m21 - m12) / S;
//     qx = 0.25 * S;
//     qy = (m01 + m10) / S;
//     qz = (m02 + m20) / S;
// }else if (m11 > m22){
//     double S = sqrt(1.0 + m11 - m00 - m22) * 2; // S=4*qy
//     qw = (m02 - m20) / S;
//     qx = (m01 + m10) / S;
//     qy = 0.25 * S;
//     qz = (m12 + m21) / S;
// }else{
//     double S = sqrt(1.0 + m22 - m00 - m11) * 2; // S=4*qz
//     qw = (m10 - m01) / S;
//     qx = (m02 + m20) / S;
//     qy = (m12 + m21) / S;
//     qz = 0.25 * S;
// }
//   return Eigen::Vector4f(qw, qx, qy, qz);
// }
// Eigen::Matrix3f qToMat(Eigen::Vector4f q) {
//
// }

Eigen::Matrix4f brac(Eigen::Matrix<float, 6, 1> x) {
  Eigen::Matrix4f S = Eigen::Matrix4f::Zero();
  Eigen::Matrix3f tmp;
  tmp << 0, -x(2), x(1), x(2), 0, -x(0), -x(1), x(0), 0;
  S.block<3, 3>(0, 0) = tmp;
  S.block<3, 1>(0, 3) = Eigen::Vector3f(x(3), x(4), x(5));
  return S;
}

Eigen::Vector3f qRot(Eigen::Vector4f q, Eigen::Vector3f v) {
  Eigen::Vector3f u = q.block<3, 1>(0, 0);
  float s = q(3);
  return (2 * u.dot(v) * u + (s * s - u.dot(u)) * v + 2 * s * u.cross(v));
}
Eigen::Vector3f qRotInv(Eigen::Vector4f q, Eigen::Vector3f v) {
  Eigen::Vector3f u = -q.block<3, 1>(0, 0);
  float s = q(3);
  return (2 * u.dot(v) * u + (s * s - u.dot(u)) * v + 2 * s * u.cross(v));
}

Eigen::Quaternionf invert_q(const Eigen::Quaternionf &q) {
  // Eigen::Vector3f tmp = -q.vec();
  // return Eigen::Quaternionf(q.w(), tmp(0), tmp(1), tmp(2));
  return Eigen::Quaternionf(q.w(), -q.x(), -q.y(), -q.z());
}

Eigen::Matrix<float, 6, 1> inertiaCuboid(Eigen::Vector3f whd, float density) {

  Eigen::Matrix<float, 6, 1> m = Eigen::Matrix<float, 6, 1>::Zero();
  float mass = density * whd.prod();
  m(0) = (1. / 12.) * mass * Eigen::Vector2f(whd(1), whd(2)).transpose() *
         Eigen::Vector2f(whd(1), whd(2));
  m(1) = (1. / 12.) * mass * Eigen::Vector2f(whd(2), whd(0)).transpose() *
         Eigen::Vector2f(whd(2), whd(0));
  m(2) = (1. / 12.) * mass * Eigen::Vector2f(whd(0), whd(1)).transpose() *
         Eigen::Vector2f(whd(0), whd(1));
  m(3) = mass;
  m(4) = mass;
  m(5) = mass;
  return m;
}
Eigen::Vector4f qMul(Eigen::Vector4f q1, Eigen::Vector4f q2) {
  // q = q1 * q2
  Eigen::Vector3f v1 = q1.block<3, 1>(0, 0);
  Eigen::Vector3f v2 = q2.block<3, 1>(0, 0);
  float r1 = q1(3);
  float r2 = q2(3);
  Eigen::Vector4f q;
  q.block<3, 1>(0, 0) = r1 * v2 + r2 * v1 + v1.cross(v2);
  q(3) = r1 * r2 - v1.dot(v2);
  return q;
}

Eigen::Vector3f cross(Eigen::Vector3f v1, Eigen::Vector3f v2) {
  Eigen::Vector3f v = Eigen::Vector3f::Zero();
  float v1x = v1(0);
  float v1y = v1(1);
  float v1z = v1(2);
  float v2x = v2(0);
  float v2y = v2(1);
  float v2z = v2(2);
  float x = v1y * v2y - v1z * v2y;
  float y = v2x * v1z - v2z * v1x;
  v(2) = v1x * v2y - v1y * v2x;
  v(0) = x;
  v(1) = y;
  return v;
}

} // namespace se3
