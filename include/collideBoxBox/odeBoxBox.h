#ifndef _ODEBOXBOX_
#define _ODEBOXBOX_

#include <Eigen/Dense>

struct Contacts {
  // Number of contacts
  int count;
  // Maximum penetration depth
  double depthMax;
  // Penetration depths
  double depths[8];
  // Contact points in world space
  Eigen::Vector3f positions[8];
  // Contact normal (same for all points)
  Eigen::Vector3f normal;
};

__host__ __device__ Contacts odeBoxBox(const Eigen::Matrix4f &M1,
                                       const Eigen::Vector3f &dimensions1,
                                       const Eigen::Matrix4f &M2,
                                       const Eigen::Vector3f &dimensions2);

#endif
