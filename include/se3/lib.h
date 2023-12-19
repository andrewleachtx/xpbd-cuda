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
} // namespace se3
