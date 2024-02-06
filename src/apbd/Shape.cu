#include "apbd/Shape.h"

namespace apbd {

Shape::Shape(ShapeCuboid cuboid) : type(SHAPE_CUBOID), data{.cuboid = cuboid} {}

__host__ __device__ bool Shape::broadphaseGround(Eigen::Matrix4f E,
                                                 Eigen::Matrix4f Eg) {
  switch (type) {
  case SHAPE_CUBOID: {
    auto data = this->data.cuboid;

    // Check the height of the center
    Eigen::Vector4f xl = {0, 0, 0, 1};
    Eigen::Vector4f xw = E * xl;
    Eigen::Vector4f xg = Eg.colPivHouseholderQr().solve(xw);
    float r = (data.sides / 2).norm(); // dist to a corner
    return xg(3) < 1.5 * r;
  }
  default:
    return false;
  }
}
__host__ __device__ bool
Shape::broadphaseShape(Eigen::Matrix4f E1, Shape *other, Eigen::Matrix4f E2) {
  switch (type) {
  case SHAPE_CUBOID:
    switch (type) {
    case SHAPE_CUBOID:
      return this->data.cuboid.broadphaseShapeCuboid(E1, &other->data.cuboid,
                                                     E2);
    default:
      return false;
    }
  default:
    return false;
  }
}

__host__ __device__ bool
ShapeCuboid::broadphaseShapeCuboid(Eigen::Matrix4f E1, ShapeCuboid *other,
                                   Eigen::Matrix4f E2) {

  Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
  Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);
  float d = (p1 - p2).norm();
  float r1 = (this->sides / 2).norm();  // dist to a corner
  float r2 = (other->sides / 2).norm(); // dist to a corner
  return d < 1.5 * (r1 + r2);
}

} // namespace apbd
