#include "model_samples.h"
#include <exception>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using std::cout, std::endl, std::string, std::runtime_error;
typedef std::chrono::high_resolution_clock Clock;

void run_cpu_thread(apbd::Model model, int sims, int processor_count, int id) {
  for (int i = id; i < sims; i += processor_count) {
    // cout << "simulating scene: " << i << endl;
    auto collider = apbd::Collider(&model);
    model.simulate(&collider);
  }
  cout << "done: " << id << endl;
}

void cpu_run_group(apbd::Model model, int sims) {
  const auto processor_count = std::thread::hardware_concurrency();
  if (processor_count == 0) {
    throw runtime_error("Failed to detect concurrency.");
  }
  auto handles = std::vector<std::thread>();
  for (int i = 0; i < processor_count; i++) {
    handles.push_back(
        std::thread(run_cpu_thread, model, sims, processor_count, i));
  }
  for (auto &h : handles) {
    cout << "waiting..." << endl;
    h.join();
  }
}

struct MainState {
  bool visualize;
  string output_file;
  int model_id;
  unsigned long scene_count;
};

const char *HELP = "\
arguments:          \n\
  -a, --algorithm ALG  Which version of the simulation algorithm to use\n\
                       values: v, gp; default v\n\
  -o, --output FILE    An output file to write the simulated positions to\n\
  -i, --model_id ID    The model scene to use\n\
  -s, --scene-count N  The number of simulations to run\n\
  -v, --visualize      Whether to visualize the finished simulation. Only\n\
                       supported on some algorithms\n\
  -h, --help           Show this help text\n\
";

MainState parse_arguments(int argc, char *argv[]) {
  struct MainState state = {
      .visualize = false,
      .output_file = "",
      .model_id = 0,
      .scene_count = 1,
  };

  option longopts[] = {{"visualize", no_argument, NULL, 'v'},
                       {"output", required_argument, NULL, 'o'},
                       {"model_id", required_argument, NULL, 'm'},
                       {"scene-count", required_argument, NULL, 's'},
                       // {"algorithm", required_argument, NULL, 'a'},
                       {"help", no_argument, NULL, 'h'},
                       {0}};

  while (1) {
    const int opt = getopt_long(argc, argv, "hvo:m:s:a:", longopts, 0);

    if (opt == -1) {
      break;
    }
    string o;
    switch (opt) {
    case 'h':
      cout << HELP << endl;
      exit(0);
    case 'v':
      state.visualize = true;
      break;
    case 'o':
      if (optarg == NULL) {
        throw runtime_error("argument for output file is required.");
      }
      cout << "output-file: " << optarg << endl;
      state.output_file = optarg;
      break;
    case 'm':
      if (optarg == NULL) {
        break;
      }
      cout << "model_id: " << optarg << endl;
      state.model_id = std::stoi(optarg);
      break;
    case 's':
      if (optarg == NULL) {
        break;
      }
      cout << "scene-count: " << optarg << endl;
      state.scene_count = std::stoul(optarg);
      break;
    // case 'a':
    //   o = optarg;
    //   if (o == "v") {
    //     state.algorithm = VALIDATION;
    //   } else if (o == "gp") {
    //     state.algorithm = GPU_PARALLEL;
    //   } else {
    //     throw runtime_error("Unknown algorithm: " + string(optarg) +
    //                         ", try v or gp.");
    //   }
    //   break;
    case '?':
    default:
      cout << "unknown option." << endl;
    }
  }
  return state;
}

int main(int argc, char *argv[]) {
  auto state = parse_arguments(argc, argv);
  auto model = createModelSample(state.model_id);

#ifdef USE_CUDA
  // TODO
#else
  cpu_run_group(model, state.scene_count);
#endif
}
