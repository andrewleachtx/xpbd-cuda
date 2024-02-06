#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

namespace apbd {

struct ShapeCuboid {
  Eigen::Vector3f sides;

  __host__ __device__ bool broadphaseShapeCuboid(Eigen::Matrix4f E1,
                                                 ShapeCuboid *other,
                                                 Eigen::Matrix4f E2);
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
  // __host__ __device__ void narrowGround(Eigen::Matrix4f E);
  __host__ __device__ bool broadphaseShape(Eigen::Matrix4f E1, Shape *other,
                                           Eigen::Matrix4f E2);
  // __host__ __device__ void narrowShape(Shape* other);
  Shape(ShapeCuboid cuboid);
};

} // namespace apbd
