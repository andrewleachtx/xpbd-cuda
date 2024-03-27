#pragma once
#include "BodyReference.h"
#include "Collider.h"
#include "Constraint.h"
#include "data/soa.h"
#include <cstddef>

namespace apbd {
class Collider;

/// Maximum number of shock propagation layers
const size_t MAX_LAYERS = 8;
/// Maximum number of objects in each layer
const size_t MAX_LAYER_SIZE = 4;

/**
 * A simulation model, contains all information necessary to run a single
 * simulation. Designed to be copied to each thread and modified with any
 * per-thread differences.
 */
class Model {
public:
  /// Duration of one simulation step
  float h;
  /// Duration of the entire simulation
  float tEnd;
  unsigned int substeps;
  /// A list of references to the bodies for this simulation, will be
  /// initialized when data is copied to the global store.
  BodyReference *bodies;
  size_t body_count;
  Constraint *constraints;
  size_t constraint_count;

  // layer objects used to calculate collision graph
  size_t *constraint_layers;
  size_t layer_count;
  size_t *constraint_layer_sizes;
  size_t *body_layers;
  size_t *body_layer_sizes;

  Eigen::Vector3f gravity;
  unsigned int iters;

  Eigen::Matrix4f ground_E;
  float ground_size;

  /// The number of simulation steps. Set when the model is initialized from h
  /// and tEnd
  unsigned int steps;

  __host__ __device__ void stepBDF1(unsigned int step, unsigned int substep,
                                    float hs);
  __host__ __device__ void clearBodyShockPropInfo();
  __host__ __device__ void constructConstraintGraph();
  __host__ __device__ void solveConSP(Collider *collider, float hs);
  __host__ __device__ void solveConGS(Collider *collider, float hs);
  __host__ __device__ void computeEnergies();

  /**
   * Constructs default data structures
   */
  Model();
  /**
   * Constructs a copy of the model, only duplicating data that cannot be shared
   */
  __host__ __device__ Model(const Model &other);
  /**
   * Moves the arrays allocated in this model to device storage.
   */
  void move_to_device();
  /**
   * Initializes the model objects based on configuration
   */
  __host__ __device__ void init();
  /**
   * Runs all simulations to completion
   */
  __host__ __device__ void simulate(Collider *collider);
  /**
   * Writes current state out for debugging
   */
  __host__ __device__ void write_state(unsigned int step);
  /**
   * Prints the configuration of this model to stdout
   */
  __host__ __device__ void print_config();
  /**
   * Creates the global data store object based on this model and the number of
   * scenes.
   */
  void create_store(size_t scene_count);
  /**
   * Copies the data in this model that uses SOA to the global store
   */
  __host__ __device__ void copy_data_to_store(Body *body_array);
};
} // namespace apbd
