#include "apbd/Model.h"
#include "util.h"

#include <iostream>
#include <stdexcept>

namespace apbd {

Model::Model()
    : h(1. / 30.), tEnd(1), substeps(10), bodies(nullptr), body_count(0),
      constraints(nullptr), constraint_count(0), constraint_layers(nullptr),
      layer_count(0), constraint_layer_sizes(nullptr), body_layers(nullptr),
      body_layer_sizes(nullptr), gravity(0, 0, -980), iters(1),
      ground_E(Eigen::Matrix4f::Zero()), ground_size(10), axis(), steps(0) {}

Model::Model(const Model &other)
    : h(other.h), tEnd(other.tEnd), substeps(other.substeps), bodies(nullptr),
      body_count(other.body_count), constraints(other.constraints),
      constraint_count(other.constraint_count), constraint_layers(nullptr),
      layer_count(other.layer_count), constraint_layer_sizes(nullptr),
      body_layers(nullptr), body_layer_sizes(nullptr), gravity(other.gravity),
      iters(other.iters), ground_E(other.ground_E),
      ground_size(other.ground_size), axis(other.axis), steps(other.steps) {
  cudaPointerAttributes attributes;
  CUDA_CHECK(cudaPointerGetAttributes(&attributes, other.bodies));
  size_t size = other.body_count * sizeof(Body);
  if (attributes.type == cudaMemoryTypeDevice) {
    bodies = alloc_device<Body>(other.body_count);
    memcpy_device(bodies, other.bodies, size);
  } else {
    bodies = new Body[other.body_count];
    memcpy(bodies, other.bodies, size);
  }
}

void Model::init() {
  // initialize bodies
  // if (this->bodies == nullptr) {
  //   throw std::runtime_error("Bodies not initialized");
  // }
  for (size_t i = 0; i < this->body_count; i++) {
    this->bodies[i].init();
  }
  // create constraints
  // if (this->constraints == nullptr && this->constraint_count) {
  //   throw std::runtime_error("Constraints not initialized");
  // }
  for (size_t i = 0; i < this->constraint_count; i++) {
    this->constraints[i].init();
  }
  // calculate parameters
  this->steps = ceil(this->tEnd / this->h);
  // this->hs = this->h / this->substeps;
  // this->k = 0;
  // this->ks
}

void Model::move_to_device() {
  bodies = move_array_to_device(bodies, body_count);
  constraints = move_array_to_device(constraints, constraint_count);
  // TODO: layers
}

void Model::simulate(Collider *collider) {
  float time = 0;
  float hs = this->h / static_cast<float>(this->substeps);
  // std::cout << this->steps << ", " << this->substeps << std::endl;
  for (unsigned int step = 0; step < this->steps; step++) {
    this->clearBodyShockPropInfo();
    collider->run(this);
    this->constructConstraintGraph();
    for (unsigned int substep = 0; substep < this->substeps; substep++) {
      this->stepBDF1(step, substep, hs);
      this->solveConSP(hs);
      this->solveConGS(hs);
      time += hs;
    }
    this->computeEnergies();
    // draw ?
  }
}

/** Private Functions **/

void Model::stepBDF1(unsigned int step, unsigned int substep, float hs) {
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].stepBDF1(step, substep, hs, this->gravity);
  }
}
void Model::clearBodyShockPropInfo() {
  // TODO
  // clears the shock propagation info from each body; this may not be necessary
  // depending on implementation
  for (size_t body_i = 0; body_i < this->body_count; body_i++) {
    this->bodies[body_i].clearShock();
  }
}
void Model::constructConstraintGraph() {
  // TODO
  // Constructs a graph of constraints, working from the ground layer up
  // needs a list of constraints and bodies
  // constraints needs:
  //  - list of bodies
  // body needs:
  //  - layer
  //  - shock parent constraint
  //  - constraint
  //
  //  collect static constraints and collision constraints
  //  for each constraint:
  //    if it is ground, initialize the body affected to layer 1 and add this
  //    constraint to the list affecting that body if the constraint has 2
  //    bodies (not 1), then tell both bodies that this constraint affects them
  //
  //  working up one layer at a time:
  //    for every body affected in the previous layer:
  //      for each constraint affecting the body:
  //        make sure the body on a higher layer is second
  //        assign the second body to this layer, and add this constraint to the
  //        parent constraints add the second body and this constraint to this
  //        layer

  // in theory, we can do this by keeping a constraint and body layer list, and
  // a last layer constraint list. the shock parent list is more difficult, but
  // does not seem to be used for anything other than setting shockProp to true;
  // so we can do this here
}
void Model::solveConSP(float hs) {
  for (size_t constraint_i = 0; constraint_i < this->constraint_count;
       constraint_i++) {
    this->constraints[constraint_i].clear();
  }

  for (size_t i = 0; i < this->layer_count; i++) {
    for (int iter = 0; iter < this->iters; iter++) {
      for (size_t j = 0; j < this->constraint_layer_sizes[i]; j++) {
        this->constraints[this->constraint_layers[i * MAX_LAYER_SIZE + j]]
            .solve(hs, true);
      }
    }
  }

  for (long i = this->layer_count - 1; i >= 0; i--) {
    for (size_t j = 0; j < this->body_layer_sizes[i]; j++) {
      this->bodies[this->body_layers[i * MAX_LAYER_SIZE + j]]
          .applyJacobiShock();
    }
    for (int iter = 0; iter < this->iters; iter++) {
      for (size_t j = 0; j < this->constraint_layer_sizes[i]; j++) {
        this->constraints[this->constraint_layers[i * MAX_LAYER_SIZE + j]]
            .solve(hs, false);
      }
    }
  }
}
void Model::solveConGS(float hs) {
  for (int iter = 0; iter < this->iters; iter++) {
    for (size_t constraint_i = 0; constraint_i < this->constraint_count;
         constraint_i++) {
      this->constraints[constraint_i].solve(hs, false);
    }
  }
}
void Model::computeEnergies() {}

} // namespace apbd
