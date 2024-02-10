#pragma once
#include "Body.h"
#include "Collider.h"
#include "Constraint.h"
#include <cstddef>

class Collider;

namespace apbd {
const size_t MAX_LAYERS = 8;
const size_t MAX_LAYER_SIZE = 4;

class Model {
public:
  // private members
  /**
   * The duration of one simulation step
   */
  float h;
  /**
   * The duration of the entire simulation
   */
  float tEnd;
  unsigned int substeps;
  Body *bodies;
  size_t body_count;
  Constraint *constraints;
  size_t constraint_count;
  size_t *constraint_layers;
  size_t layer_count;
  size_t *constraint_layer_sizes;
  size_t *body_layers;
  size_t *body_layer_sizes;
  Eigen::Vector3f gravity;
  unsigned int iters;

  Eigen::Matrix4f ground_E;
  float ground_size;
  Eigen::Matrix<float, 6, 1> axis;
  Collider *collider;

  // derived
  unsigned int steps;

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs);
  __host__ __device__ void clearBodyShockPropInfo();
  __host__ __device__ void constructConstraintGraph();
  __host__ __device__ void solveConSP(float hs);
  __host__ __device__ void solveConGS(float hs);
  __host__ __device__ void computeEnergies();

  /**
   * constructs default data structures
   */
  Model();
  /**
   * Initializes the model objects based on configuration
   */
  __host__ __device__ void init(/* TODO */);
  /**
   * Runs all simulations to completion
   */
  __host__ __device__ void simulate();
};
} // namespace apbd
