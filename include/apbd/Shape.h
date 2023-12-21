#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

namespace apbd {

struct ShapeCuboid {
  Eigen::Vector3f sides;
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

  Shape(ShapeCuboid cuboid);
};

} // namespace apbd
