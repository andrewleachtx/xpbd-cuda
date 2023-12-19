#include "apbd/Shape.h"

namespace apbd {

Shape::Shape(ShapeCuboid cuboid) : type(SHAPE_CUBOID), data{.cuboid = cuboid} {}

} // namespace apbd
