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

void Collider::narrowphase() {}
} // namespace apbd
