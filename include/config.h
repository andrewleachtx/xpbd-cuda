#pragma once
#include <stddef.h>

/// Maximum number of collisions within the entire simulation
const size_t MAX_COLLISIONS = 1024;
/// Block size used in CUDA kernel
const size_t BLOCK_SIZE = 256;
