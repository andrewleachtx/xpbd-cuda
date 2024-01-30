#include <apbd/Constraint.h>
#include <se3/lib.h>

using Eigen::Vector2f, Eigen::Vector3f, Eigen::Vector4f;

namespace apbd {

void Constraint::clear()
    {
      switch (type)
      {
      case CONSTRAINT_COLLISION_GROUND: {
        ConstraintGround *c = &data.ground;
        c->C = Eigen::Vector3f::Zero();
        c->lambda = Eigen::Vector3f::Zero();
        break;}
      case CONSTRAINT_COLLISION_RIGID:{
        ConstraintRigid *c = &data.rigid;
        c->C = Eigen::Vector3f::Zero();
        c->lambda = Eigen::Vector3f::Zero();
        break;}
      case CONSTRAINT_JOINT_REVOLVE:{
        ConstraintJointRevolve *c = &data.joint;
        c->C = Eigen::Vector3f::Zero();
        c->lambda = Eigen::Vector3f::Zero();
        break;}

      default:
        break;
      }
    }
    
      void Constraint::solve(float hs, bool doShockProp)
    {
      switch (type)
      {
      case CONSTRAINT_COLLISION_GROUND:{
        ConstraintGround *c = &data.ground;
        c->solveNorPos(hs);
        c->applyJacobi();
        break;}
      case CONSTRAINT_COLLISION_RIGID:{
        ConstraintRigid *c = &data.rigid;
        c->solveNorPos(hs);
        c->applyJacobi();
        break;}
      case CONSTRAINT_JOINT_REVOLVE:{
        ConstraintJointRevolve *c = &data.joint;
        c->solve();
        break;}

      default:
        break;
      }
    }

    void ConstraintGround::solveNorPos(float hs) {
      Vector3f v = hs * body->computePointVel(xl, hs);
      float vNorm = v.norm();
      Vector3f vNormalized = v / vNorm;
      Vector3f tx = Eg.block<3, 1>(0, 0);
      Vector3f ty = Eg.block<3, 1>(0, 1);
      Eigen::Matrix3f frame_tmp;
      frame_tmp << nw, tx, ty;
      Vector3f vNormalizedContactFrame = frame_tmp * vNormalized;
  
      float dlambda = solvePosDir1(vNorm, vNormalized);
      C = vNorm * vNormalizedContactFrame;
  
      float dlambdaNor = dlambda * vNormalizedContactFrame(0);
      float lambdaNor = lambda(1) + dlambdaNor;
      if (lambdaNor < 0) {
          dlambdaNor = -lambda(1);
      }
      lambda(1) += dlambdaNor;
      float mu = body->mu;
      Vector3f dlambdaTan;
      if (mu > 0) {
          float dlambdaTx = dlambda * vNormalizedContactFrame(1);
          float dlambdaTy = dlambda * vNormalizedContactFrame(2);
          float lambdaNorLenMu = mu * lambda(1);
          Vector2f lambdaTan = Vector2f(lambda(2) + dlambdaTx, lambda(3) + dlambdaTy);
          float lambdaTanLen = lambdaTan.norm();
          auto dlambdaTan = Vector2f(dlambdaTx, dlambdaTy);
          if (lambdaTanLen > lambdaNorLenMu) {
          dlambdaTan = (lambdaTan / lambdaTanLen * lambdaNorLenMu - Vector2f(lambda(2), lambda(3)));}
          lambda(2) += dlambdaTan(0);
          lambda(3) += dlambdaTan(1);
      }
  
      Vector3f frictionalContactLambda = Vector3f(dlambdaNor, 0, 0) + dlambdaTan;
      dlambda = frictionalContactLambda.norm();
      if (dlambda > 0) {
        // frictionalContactNormal = [this.nw, tx, ty] * frictionalContactLambda ./ dlambda;
          Eigen::Matrix3f tmp;
          tmp << nw, tx, ty;
          Vector3f frictionalContactNormal = (tmp * frictionalContactLambda).array() / dlambda;
          vec7 dq = computeDx(dlambda, frictionalContactNormal);
          body->dxJacobi.block<4, 1>(0, 0) += dq.block<4, 1>(0, 0);
          body->dxJacobi.block<3, 1>(4, 0) += dq.block<3, 1>(4, 0);
      }
  }

  vec7 ConstraintGround::computeDx(float dlambda, Eigen::Vector3f frictionalContactNormal) {

			float m1 = body->Mp;
       Vector3f I1 = body->Mr;
			// Position update
			Vector3f dpw = dlambda*nw;
			Vector3f dp = dpw/m1;
			// Quaternion update
      Vector4f q1 = body->x1_0.block<1,4>(0,0);
			auto dpl1 = se3::qRotInv(q1,dpw);
			Vector4f qtmp1;
      qtmp1 << se3::qRot(q1, xl.cross(dpl1).array() / I1.array()), 0;
            //qtmp1 = [I1.\se3.cross(rl1,dpl1); 0];
			//dq = se3.qMul(sin(0.5*qtmp1),q1);
      Vector4f dq = 0.5 * se3::qMul(qtmp1,q1);
      vec7 out;
      out << dq, dp;
      return out;
  }
}