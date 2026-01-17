#include <iostream>
#include <filesystem>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>

namespace fs = std::filesystem;

int main() {
    std::cout << "Creating the Sim" << std::endl;

    // Create checkpoint directory
    std::string checkpoint_dir = "/workspaces/inverse_pendolum_training/checkpoints";
    if (!fs::exists(checkpoint_dir)) {
        fs::create_directories(checkpoint_dir);
        std::cout << "Created checkpoint directory: " << checkpoint_dir << std::endl;
    }

    std::string model_path = "/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml";

    mj_pool::Sim sim;
    sim.create_model(model_path.c_str()); // This function should be called internally in the init method. It should load all the parameters
    // from a configuration file
    sim.init();   

    // // Set Torch to single-threaded execution for interference to avoid OpenMP conflicts
    // torch::set_num_threads(1);

    // Setup controller with access to all Sim buffers
    int num_envs = 1000; // Matches Sim default
    int n_cores = omp_get_num_procs();
    
    auto controller = std::make_shared<rl::Controller>(
        sim.get_action_buffer(),
        sim.get_observation_buffer(),
        sim.get_log_prob_buffer(),
        sim.get_value_buffer(),
        num_envs,
        n_cores
    );
    
    controller->init();
    sim.set_controller(controller);

    // Run training loop
    int num_iterations = 25;
    int steps_per_iteration = 1000;

    for (int i = 0; i < num_iterations; ++i) {
        std::cout << "Iteration " << i + 1 << "/" << num_iterations << " started..." << std::endl;
        sim.run(steps_per_iteration);

        if ((i + 1) % 5 == 0) {
            controller->save(checkpoint_dir, i + 1);
        }
    }
   
    return 0;
}