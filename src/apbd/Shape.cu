#include "apbd/Shape.h"
#include "collideBoxBox/odeBoxBox.h"
#include "JGT-float/smits_mul.h"

namespace apbd {

Shape::Shape(ShapeCuboid cuboid) : type(SHAPE_CUBOID), data{.cuboid = cuboid} {}

bool Shape::broadphaseGround(Eigen::Matrix4f E, Eigen::Matrix4f Eg) {
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

cuda::std::pair<cuda::std::array<CollisionGround, 8>, size_t>
Shape::narrowphaseGround(Eigen::Matrix4f E, Eigen::Matrix4f Eg) {
  auto cdata = cuda::std::array<CollisionGround, 8>();
  switch (type) {
  case SHAPE_CUBOID: {
    auto data = this->data.cuboid;

    Eigen::Vector3f s = data.sides / 2;
    // local space
    Eigen::Matrix<float, 4, 8> xl = Eigen::Matrix<float, 4, 8>::Ones();
    xl.block<3, 1>(0, 0) = Eigen::Vector3f(-s(1), -s(2), -s(3));
    xl.block<3, 1>(0, 1) = Eigen::Vector3f(s(1), -s(2), -s(3));
    xl.block<3, 1>(0, 2) = Eigen::Vector3f(s(1), s(2), -s(3));
    xl.block<3, 1>(0, 3) = Eigen::Vector3f(-s(1), s(2), -s(3));
    xl.block<3, 1>(0, 4) = Eigen::Vector3f(-s(1), -s(2), s(3));
    xl.block<3, 1>(0, 5) = Eigen::Vector3f(s(1), -s(2), s(3));
    xl.block<3, 1>(0, 6) = Eigen::Vector3f(s(1), s(2), s(3));
    xl.block<3, 1>(0, 7) = Eigen::Vector3f(-s(1), s(2), s(3));

    Eigen::Matrix<float, 4, 8> xw = E * xl;
    Eigen::Matrix<float, 4, 8> xg = Eg.colPivHouseholderQr().solve(xw);

    int cdata_count = 0;
    for (size_t i = 0; i < 8; i++) {
      // This only supports vertex collisions
      float d = xg(2, i);
      if (d < 0) {
        Eigen::Vector4f xgproj = xg.block<1, 4>(0, i);
        // project onto the floor plane
        xgproj(2) = 0;
        cdata[cdata_count++] = CollisionGround{
            .d = d,
            .xl = xl.block<1, 3>(0, i),
            // transform to world space
            .xw = Eg.block<3, 4>(0, 0) * xgproj,
            // normal
            .nw = Eg.block<3, 1>(0, 2),
            // unused for now, assuming the ground
            .vw = Eigen::Vector3f::Zero(),
        };
      }
    }

    return cuda::std::pair(cdata, cdata_count);
  }
  default:
    return cuda::std::pair(cdata, 0);
  }
}

bool Shape::broadphaseShape(Eigen::Matrix4f E1, Shape *other,
                            Eigen::Matrix4f E2) {
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

cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
Shape::narrowphaseShape(Eigen::Matrix4f E1, Shape *other, Eigen::Matrix4f E2) {
  switch (type) {
  case SHAPE_CUBOID:
    switch (type) {
    case SHAPE_CUBOID:
      return this->data.cuboid.narrowphaseShapeCuboid(E1, &other->data.cuboid,
                                                      E2);
    default:
      return cuda::std::pair(cuda::std::array<CollisionRigid, 8>(), 0);
    }
  default:
    return cuda::std::pair(cuda::std::array<CollisionRigid, 8>(), 0);
  }
}

bool ShapeCuboid::broadphaseShapeCuboid(Eigen::Matrix4f E1, ShapeCuboid *other,
                                        Eigen::Matrix4f E2) {

  Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
  Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);
  float d = (p1 - p2).norm();
  float r1 = (this->sides / 2).norm();  // dist to a corner
  float r2 = (other->sides / 2).norm(); // dist to a corner
  return d < 1.5 * (r1 + r2);
}

cuda::std::pair<cuda::std::array<CollisionRigid, 8>, size_t>
ShapeCuboid::narrowphaseShapeCuboid(Eigen::Matrix4f E1, ShapeCuboid *other,
                                    Eigen::Matrix4f E2) {
  cuda::std::array<CollisionRigid, 8> cdata{};
  Eigen::Matrix3f R1 = E1.block<3, 3>(0, 0);
  Eigen::Matrix3f R2 = E2.block<3, 3>(0, 0);
  Eigen::Vector3f p1 = E1.block<3, 1>(0, 3);
  Eigen::Vector3f p2 = E2.block<3, 1>(0, 3);
  auto &s1 = this->sides;
  auto &s2 = other->sides;
  auto collisions = odeBoxBox(E1, s1, E2, s2);
  Eigen::Vector3f nw =
      collisions.normal; // The normal is outward from body 1 (red)
  Eigen::Vector3f n1 = R1.transpose() * nw;
  Eigen::Vector3f n2 =
      -R2.transpose() * nw; // negate since nw is defined wrt body 1
  for (size_t i = 0; i < collisions.count && i < 8; i++) {
    Eigen::Vector3f xw = collisions.positions[i];
    float d =
        -collisions.depths[i]; // odeBoxBox returns positive depth for hits
                               // Compute local point on body 1 with ray casting
    Eigen::Vector3f x1 = R1.transpose() * (xw - p1);
    float t1 = this->raycast(x1, n1);

    // negate since smits_mul returns negative t for rays starting
    // inside the box Compute local point on body 2 with ray casting
    x1 = x1 - t1 * n1;

    Eigen::Vector3f x2 = R2.transpose() * (xw - p2);
    float t2 = other->raycast(x2, n2);
    x2 = x2 - t2 * n2; // negate since smits_mul returns negative t for rays
                       // starting inside the box
    cdata[i] = CollisionRigid{.d = d, .xw = xw, .nw = nw, .x1 = x1, .x2 = x2};
  }
  return cuda::std::pair(cdata, collisions.count);
}

float ShapeCuboid::raycast(Eigen::Vector3f x, Eigen::Vector3f n) {
  float thresh = 1e-6;
  Eigen::Vector3f bmax = 0.5 * this->sides;
  Eigen::Vector3f bmin = -bmax;
  x = (1 - thresh) * x; // make the point go slightly inside the box
  n = -n; // negate ray since it starts inside the box
  jgt_float::ray r;
  jgt_float::make_ray(x(1), x(2), x(3), n(1), n(2), n(3), &r);
  jgt_float::aabox a;
  jgt_float::make_aabox(bmin(1), bmin(2), bmin(3), bmax(1), bmax(2), bmax(3), &a);
  float t;
  bool _hit = jgt_float::smits_mul(&r, &a, &t);
  return t;
}

} // namespace apbd
