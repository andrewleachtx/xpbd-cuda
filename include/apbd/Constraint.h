#pragma once

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#include <Eigen/Dense>
#include <apbd/Body.h>

namespace apbd
{

  enum CONSTRAINT_TYPE
  {
    CONSTRAINT_COLLISION_GROUND,
    CONSTRAINT_COLLISION_RIGID,
    CONSTRAINT_JOINT_REVOLVE,
  };

  struct ConstraintGround
  {
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

    vec7 computeDx(float dlambda, Eigen::Vector3f frictionalContactNormal);
    float solvePosDir1(float c, Eigen::Vector3f nw);
    void solveNorPos(float hs);
    void applyJacobi();
  };

  struct ConstraintRigid
  {
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

    void solveNorPos(float hs);
    void applyJacobi();
  };

  struct ConstraintJointRevolve
  {
    Eigen::Vector3f C;
    Eigen::Vector3f lambda;
    bool shockProp;
    BodyRigid *body1;
    BodyRigid *body2;
    Eigen::Vector4f ql1;
    Eigen::Vector4f pl1;
    Eigen::Vector4f ql2;
    Eigen::Vector4f pl2;

    void solve();
  };

  union ConstraintInner
  {
    ConstraintGround ground;
    ConstraintRigid rigid;
    ConstraintJointRevolve joint;
  };

  class Constraint
  {
  public:
    CONSTRAINT_TYPE type;
    ConstraintInner data;

    __host__ __device__ void clear();

    __host__ __device__ void solve(float hs, bool doShockProp);
  };

} // namespace apbd
