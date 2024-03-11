#include "model_samples.h"

int main(int argc, char **argv) {
  if (argc < 2)
    exit(1);
  apbd::Model model = createModelSample(atoi(argv[1]));

  auto collider = apbd::Collider(&model);
  model.simulate(&collider);
}
