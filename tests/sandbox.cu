#include "apbd/BodyReference.h"
#include "model_samples.h"
#include <exception>
#include <getopt.h>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

using std::cout, std::endl, std::string, std::runtime_error;
typedef std::chrono::high_resolution_clock Clock;

__global__ void kernel(apbd::Model model, apbd::Body *body_buffer,
                       apbd::Body **body_ptr_buffer,
                       apbd::Constraint *constraint_buffer, int sims) {
  // get this scene ID
  size_t scene_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (scene_id >= sims)
    return;
  model.copy_data_to_store();
  auto r = apbd::BodyReference(0, apbd::BODY_RIGID);
  auto b = r.get_rigid();

  DEBUG_VEC(b.xdotInit(), 7);

  // // make a copy of the model
  // apbd::Model thread_model =
  //     apbd::Model::clone_with_buffers(model, scene_id, body_buffer);
  // // create a thread-local collider
  // auto collider = apbd::Collider(&thread_model, scene_id, body_ptr_buffer,
  //                                constraint_buffer);
  // // simulate
  // thread_model.simulate(&collider);
}

void run_kernel(apbd::Model model, int sims) {
  cout << "thread blocks: " << (sims + BLOCK_SIZE - 1) / BLOCK_SIZE << endl;

  // const size_t shared_size = c.constraints.size() * sizeof(Constraint);
  const size_t shared_size = 0;
  // std::cout << "kernel shared_size: " << shared_size << " = " <<
  // c.constraints.size() << " * " << sizeof(Constraint) << std::endl;

  size_t body_buffer_size = sims * model.body_count;
  apbd::Body *body_buffer = alloc_device<apbd::Body>(body_buffer_size);
  apbd::Body **body_ptr_buffer = nullptr;
  apbd::Constraint *constraint_buffer = nullptr;
  apbd::Collider::allocate_buffers(model, sims, body_ptr_buffer,
                                   constraint_buffer);

  model.move_to_device();

  std::cout << "kernel start" << std::endl;
  auto t1 = Clock::now();

  kernel<<<(sims + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, shared_size>>>(
      model, body_buffer, body_ptr_buffer, constraint_buffer, sims);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto t2 = Clock::now();
  std::cout << "Kernel took: " << (t2 - t1).count() << '\t';
}

int main(int argc, char *argv[]) {
  // auto state = parse_arguments(argc, argv);
  auto model = createModelSample(atoi(argv[1]));
  model.create_store();

  // auto t1 = Clock::now();
#ifdef USE_CUDA
  cout << "Running with CUDA" << endl;
  run_kernel(model, 1);
#else
  // cout << "Running on CPU" << endl;
  // cpu_run_group(model, state.scene_count);
#endif
  // auto t2 = Clock::now();
  // cout << "Simulation took: " << (t2 - t1).count() << '\n';
}
