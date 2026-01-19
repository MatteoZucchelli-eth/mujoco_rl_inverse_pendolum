#include <iostream>
#include <filesystem>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::cout << "Creating the Sim" << std::endl;

    // Create checkpoint directory
    std::string checkpoint_dir = "/workspaces/inverse_pendolum_training/checkpoints_6";
    if (!fs::exists(checkpoint_dir)) {
        fs::create_directories(checkpoint_dir);
        std::cout << "Created checkpoint directory: " << checkpoint_dir << std::endl;
    }

    std::string model_path = "/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml";

    mj_pool::Sim sim;
    sim.init(model_path.c_str());   

    // Setup controller with access to all Sim buffers
    int num_envs = 104;
    int n_cores = omp_get_num_procs();
    
    auto controller = std::make_shared<rl::Controller>(
        sim.get_action_buffer(),
        sim.get_observation_buffer(),
        sim.get_log_prob_buffer(),
        sim.get_value_buffer(), // Here I think I should give to it the advantage buffer, not the value buffer
        num_envs,
        n_cores
    );
    
    controller->init();
    
    int start_iteration = 0;
    if (argc > 1) {
        std::string checkpoint_path = argv[1];
        std::cout << "Loading checkpoint from: " << checkpoint_path << std::endl;
        controller->load(checkpoint_path);
        
        // Try to parse iteration
        try {
            size_t pos = checkpoint_path.find("actor_");
            if (pos != std::string::npos) {
                size_t start = pos + 6;
                size_t end = checkpoint_path.find(".pt", start);
                if (end != std::string::npos) {
                    std::string num_str = checkpoint_path.substr(start, end - start);
                    start_iteration = std::stoi(num_str);
                    std::cout << "Resuming from iteration " << start_iteration << std::endl;
                }
            }
        } catch (...) {
            std::cout << "Could not parse iteration number from filename." << std::endl;
        }
    }

    sim.set_controller(controller);

    // Run training loop
    int num_iterations = 200;
    int steps_per_iteration = 2500;

    for (int i = start_iteration; i < start_iteration + num_iterations; ++i) {
        std::cout << "Iteration " << i + 1 << "/" << (start_iteration + num_iterations) << " started..." << std::endl;
        sim.run(steps_per_iteration);

        if ((i + 1) % 10 == 0) {
            controller->save(checkpoint_dir, i + 1);
        }
    }
   
    return 0;
}