#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <apbd/Body.h>

namespace apbd {

enum CONSTRAINT_TYPE {
  CONSTRAINT_COLLISION_GROUND,
  CONSTRAINT_COLLISION_RIGID,
  CONSTRAINT_JOINT_REVOLVE,
};

struct ConstraintGround {
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f lambdaSF;
  float d;
  float dlambdaNor;
  bool shockProp;
  BodyRigid *body;
  Eigen::Matrix4f Eg;
  Eigen::Vector3f xl;
  Eigen::Vector3f xw;
  Eigen::Vector3f vw;

  __host__ __device__ ConstraintGround(BodyRigid *body, Eigen::Matrix4f Eg,
                                       float d, Eigen::Vector3f xl,
                                       Eigen::Vector3f xw, Eigen::Vector3f nw,
                                       Eigen::Vector3f vw);

  __host__ __device__ vec7 computeDx(float dlambda,
                                     Eigen::Vector3f frictionalContactNormal);
  __host__ __device__ float solvePosDir1(float c, Eigen::Vector3f nw);
  __host__ __device__ void solveNorPos(float hs);
  __host__ __device__ void applyJacobi();
};

struct ConstraintRigid {
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  Eigen::Vector3f nw;
  Eigen::Vector3f lambdaSF;
  float d;
  float dlambdaNor;
  bool shockProp;
  BodyRigid *body1;
  BodyRigid *body2;
  Eigen::Vector3f x1;
  Eigen::Vector3f x2;

  __host__ __device__ ConstraintRigid(BodyRigid *body1, BodyRigid *body2,
                                      float d, Eigen::Vector3f nw,
                                      Eigen::Vector3f x1, Eigen::Vector3f x2);

  __host__ __device__ void solveNorPos(float hs);
  __host__ __device__ float solvePosDir2(float c, Eigen::Vector3f nw);
  __host__ __device__ void computeDx(float dlambda, Eigen::Vector3f nw,
                                     Eigen::Vector4f *dq1, Eigen::Vector3f *dp1,
                                     Eigen::Vector4f *dq2,
                                     Eigen::Vector3f *dp2);
  __host__ __device__ void applyJacobi();
};

struct ConstraintJointRevolve {
  Eigen::Vector3f C;
  Eigen::Vector3f lambda;
  bool shockProp;
  BodyRigid *body1;
  BodyRigid *body2;
  Eigen::Vector4f ql1;
  Eigen::Vector4f pl1;
  Eigen::Vector4f ql2;
  Eigen::Vector4f pl2;

  __host__ __device__ void solve();
};

union ConstraintInner {
  ConstraintGround ground;
  ConstraintRigid rigid;
  ConstraintJointRevolve joint_revolve;
};

class Constraint {
public:
  CONSTRAINT_TYPE type;
  ConstraintInner data;

  __host__ __device__ Constraint(ConstraintRigid rigid);
  __host__ __device__ Constraint(ConstraintGround ground);
  __host__ __device__ Constraint(ConstraintJointRevolve revolve);
  __host__ __device__ Constraint &operator=(const Constraint &);

  __host__ __device__ void init();

  __host__ __device__ void clear();

  __host__ __device__ void solve(float hs, bool doShockProp);
};

} // namespace apbd
