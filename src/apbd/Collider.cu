#include "apbd/Collider.h"
#include "config.h"

namespace apbd {

const size_t MAX_COLLISIONS = 64;

Collider::Collider(Model *model)
    : bp_cap_1(model->body_count), bp_cap_2(model->body_count * 2),
      bp_count_1(0), bp_count_2(0), bpList1(nullptr), bpList2(nullptr),
      collision_count(0), collisions(nullptr) {
  // TODO: allocate bpLists on device/host
#ifdef USE_CUDA
  // TODO
#else
  bpList1 = new Body *[bp_cap_1];
  bpList2 = new Body *[bp_cap_2];
  collisions = reinterpret_cast<Constraint *>(
      new char[MAX_COLLISIONS * sizeof(Constraint)]);
#endif
}

void Collider::run(Model *model) {
  bp_count_1 = 0;
  bp_count_2 = 0;
  collision_count = 0;
  this->broadphase(model);
  this->narrowphase(model);
}

void Collider::broadphase(Model *model) {
  Body *bodies = model->bodies;

  for (size_t i = 0; i < model->body_count && this->bp_count_1 < this->bp_cap_1;
       i++) {
    Body *body = &bodies[i];
    if (body->collide()) {
      if (body->broadphaseGround(model->ground_E)) {
        this->bpList1[this->bp_count_1++] = body;
      }
    }
  }
  for (size_t i = 0; i < model->body_count && this->bp_count_2 < this->bp_cap_2;
       i++) {
    Body *body = &bodies[i];
    if (body->collide()) {
      for (size_t j = i + 1;
           j < model->body_count && this->bp_count_2 < this->bp_cap_2; j++) {
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

void Collider::narrowphase(Model *model) {
  auto Eg = model->ground_E;

  for (size_t i = 0; i < this->bp_count_1; i++) {
    auto body = this->bpList1[i];
    auto cpair = body->narrowphaseGround(Eg);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 0; k < c_count && this->collision_count < MAX_COLLISIONS;
         k++) {
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

  for (size_t i = 0; i < this->bp_count_2; i += 2) {
    auto body1 = this->bpList2[i];
    auto body2 = this->bpList2[i + 1];
    auto cpair = body1->narrowphaseRigid(body2);
    auto cdata = cpair.first;
    auto c_count = cpair.second;
    for (size_t k = 0; k < c_count; k++) {
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
