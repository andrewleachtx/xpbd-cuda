#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include "Collisions.h"
#include <Eigen/Dense>
#include <cuda/std/array>
#include <cuda/std/utility>

namespace apbd {

struct ShapeCuboid {
  Eigen::Vector3f sides;

  __host__ __device__ bool broadphaseShapeCuboid(Eigen::Matrix4f E1,
                                                 ShapeCuboid *other,
                                                 Eigen::Matrix4f E2);
  __host__ __device__ cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t> 
  narrowphaseShapeCuboid(Eigen::Matrix4f E1, ShapeCuboid *other,
                         Eigen::Matrix4f E2);
  __host__ __device__ float raycast(Eigen::Vector3f x, Eigen::Vector3f n);
};

enum SHAPE_TYPE {
  SHAPE_CUBOID,
};

union _ShapeInner {
  ShapeCuboid cuboid;
};

class Shape {
public:
  SHAPE_TYPE type;
  _ShapeInner data;

  // TODO: handle other shapes
  Shape(const Shape &other)
      : type(other.type), data{.cuboid = other.data.cuboid} {
    // this->type = other.type;
    // switch (other.type) {
    //   case SHAPE_CUBOID:
    //
    // this->data.cuboid = other.data.cuboid;
    //   default:
    //     break;
    // }
  }

  __host__ __device__ bool broadphaseGround(Eigen::Matrix4f E,
                                            Eigen::Matrix4f Eg);
  __host__ __device__ cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>
  narrowphaseGround(Eigen::Matrix4f E, Eigen::Matrix4f Eg);
  __host__ __device__ bool broadphaseShape(Eigen::Matrix4f E1, Shape *other,
                                           Eigen::Matrix4f E2);
  __host__ __device__ cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
  narrowphaseShape(Eigen::Matrix4f E1, Shape *other, Eigen::Matrix4f E2);
  Shape(ShapeCuboid cuboid);
};

} // namespace apbd
