#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

namespace se3 {
const float THRESH = 1e-9;

///
__host__ __device__ Eigen::Vector3f qdotToW(Eigen::Vector4f q,
                                            Eigen::Vector4f qdot);

///
__host__ __device__ Eigen::Vector4f wToQdot(Eigen::Vector4f q,
                                            Eigen::Vector3f w);

__host__ __device__ Eigen::Matrix3f aaToMat(Eigen::Vector3f axis, float angle);

// __host__ __device__ Eigen::Vector4f matToQ(Eigen::Matrix3f R);
// __host__ __device__ Eigen::Matrix3f qToMat(Eigen::Vector4f q);

__host__ __device__ Eigen::Matrix4f brac(Eigen::Matrix<float, 6, 1> x);

__host__ __device__ Eigen::Vector3f qRot(Eigen::Vector4f q, Eigen::Vector3f v);
__host__ __device__ Eigen::Vector3f qRotInv(Eigen::Vector4f q,
                                            Eigen::Vector3f v);
__host__ __device__ Eigen::Quaternionf invert_q(const Eigen::Quaternionf &q);

__host__ __device__ Eigen::Vector4f qMul(Eigen::Vector4f q1,
                                         Eigen::Vector4f q2);

__host__ __device__ Eigen::Vector3f cross(Eigen::Vector3f v1,
                                          Eigen::Vector3f v2);
/**
 * Gets the diagonal inertia of a cuboid with (width, height, depth)
 */
__host__ __device__ Eigen::Matrix<float, 6, 1>
inertiaCuboid(Eigen::Vector3f whd, float density);
} // namespace se3
