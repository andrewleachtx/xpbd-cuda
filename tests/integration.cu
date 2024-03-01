#include "model_samples.h"

int main(int argc, char **argv) {
  apbd::Model model = createModelSample(-1);

  auto collider = apbd::Collider(&model);
  model.simulate(&collider);
}
