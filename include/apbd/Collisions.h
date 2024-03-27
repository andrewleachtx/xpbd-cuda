#pragma once
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>

// These collision objects are used to store information when collisions are
// calculated to later create collision constraints

struct CollisionGround {
  float d;
  Eigen::Vector3f xl;
  Eigen::Vector3f xw;
  Eigen::Vector3f nw;
  Eigen::Vector3f vw;
};

struct CollisionRigid {
  float d;
  Eigen::Vector3f xw;
  Eigen::Vector3f nw;
  Eigen::Vector3f x1;
  Eigen::Vector3f x2;
};
