#include "apbd/Collider.h"

namespace apbd {

Collider::Collider(Model *model)
    : model(model), bp_cap_1(model->body_count),
      bp_cap_2(model->body_count * 2), bp_count_1(0), bp_count_2(0),
      bpList1(nullptr), bpList2(nullptr), collision_count(0),
      collisions(nullptr) {
  // TODO: allocate bpLists on device/host
}

void Collider::run() {
  bp_count_1 = 0;
  bp_count_2 = 0;
  collision_count = 0;
  this->broadphase();
  this->narrowphase();
}

void Collider::broadphase() {
  Body *bodies = this->model->bodies;

  for (size_t i = 0; i < this->model->body_count; i++) {
    Body *body = &bodies[i];
    if (body->collide()) {
      if (body->broadphaseGround(this->model->ground_E)) {
        this->bpList1[this->bp_count_1++] = body;
      }
    }
  }
  for (size_t i = 0; i < this->model->body_count; i++) {
    Body *body = &bodies[i];
    if (body->collide()) {
      for (size_t j = i + 1; j < this->model->body_count; j++) {
        if (bodies[j].collide()) {
          if (body->broadphaseRigid(&bodies[j])) {
            this->bpList2[this->bp_count_2++] = body;
            this->bpList2[this->bp_count_2++] = &bodies[j];
          }
        }
      }
    }
  }
}

void Collider::narrowphase() {
  auto Eg = this->model->ground_E;

  for (size_t i = 1; i < this->bp_count_1; i++) {
    auto body = this->bpList1[i];
    auto cpair = body->narrowphaseGround(Eg);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 1; k < c_count; k++) {
      auto &c = cdata[k];
      switch (body->type) {
      case BODY_RIGID: {
        this->collisions[this->collision_count++] = Constraint(ConstraintGround(
            &body->data.rigid, Eg, c.d, c.xl, c.xw, c.nw, c.vw));
        break;
      }
      default:
        break;
      }
    }
  }

  for (size_t i = 1; i < this->bp_count_2; i += 2) {
    auto body1 = this->bpList2[i];
    auto body2 = this->bpList2[i + 1];
    auto cpair = body1->narrowphaseRigid(body2);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 1; k < c_count; k++) {
      auto &c = cdata[k];
      switch (body1->type) {
      case BODY_RIGID: {
        // we require the other body to be rigid
        if (body2->type == BODY_RIGID) {
          this->collisions[this->collision_count++] =
              Constraint(ConstraintRigid(&body1->data.rigid, &body2->data.rigid,
                                         c.d, c.nw, c.x1, c.x2));
        }
        break;
      }
      default:
        break;
      }
    }
  }
}

std::pair<Eigen::Vector3f, Eigen::Vector3f>
Collider::generateTangents(Eigen::Vector3f nor) {
  Eigen::Vector3f tmp;
  if (abs(nor(3)) < 1e-6) {
    tmp << 0, 0, 1;
  } else {
    tmp << 1, 0, 0;
  }
  Eigen::Vector3f tany = nor.cross(tmp);
  tany = tany / tany.norm();
  Eigen::Vector3f tanx = tany.cross(nor);
  return std::pair(tanx, tany);
}

} // namespace apbd
