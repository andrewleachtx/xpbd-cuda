#pragma once
namespace apbd {

enum CONSTRAINT_TYPE {
  CONSTRAINT_DISTANCE,
  CONSTRAINT_GROUND,
  CONSTRAINT_FIXED,
};

class Constraint {
public:
  CONSTRAINT_TYPE type;

  __host__ __device__ void clear() {
    // TODO
  }

  __host__ __device__ void solve(float hs, bool doShockProp) {
    // TODO
  }
};

} // namespace apbd
