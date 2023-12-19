#include "apbd/Model.h"
#include <math.h>

using Eigen::seq;

int main(int argc, char **argv) {
  auto model = apbd::Model();
  const int modelID = 0;

  switch (modelID) {
  case 0:
    // model.name = 'Rigid Body';
    // model.plotH = true;
    model.tEnd = 1;
    model.h = 1 / 30;
    model.substeps = 10;
    model.iters = 1;
    float density = 1.0;
    float l = 5;
    float w = 1;
    auto sides = Eigen::Vector3f(l, w, w);
    model.gravity = 0 * Eigen::Vector3f(0, 0, -980).transpose();
    model.ground_E = Eigen::Matrix4f::Identity();

    model.ground_size = 20;
    model.axis = 10 * Eigen::Matrix<float, 6, 1>(-1, 1, -1, 1, 0, 1);
    // model.drawHz = 30;

    model.body_count = 1;
    model.bodies = new apbd::Body[1]{
        apbd::Body(apbd::BodyRigid(apbd::ShapeCuboid{sides}, density))};

    Eigen::Matrix4f E = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f R = se3::aaToMat(Eigen::Vector3f(1, 1, 1), M_PI / 4);
    E.block<3, 3>(0, 0) = R;
    E.block<1, 3>(0, 3) = Eigen::Vector3f(0, 0, 5).transpose();
    model.bodies[0].setInitTransform(E);
    Eigen::Vector3f x1 = R.transpose() * Eigen::Vector3f(3, -4, 5);
    Eigen::Vector3f x2 = R.transpose() * Eigen::Vector3f(0, 0, 5);
    Eigen::Matrix<float, 6, 1> v;
    v << x1, x2;
    model.bodies[0].setInitVelocity(v);
    // model.bodies{end}.setInitVelocity([0 0 0 0 0 1]');
  }
  model.init();
  model.simulate();
}
