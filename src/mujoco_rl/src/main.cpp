#include <iostream>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>

int main() {
    std::cout << "Creating the Sim" << std::endl;

    std::string model_path = "/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml";

    mj_pool::Sim sim;
    sim.create_model(model_path.c_str()); // This function should be called internally in the init method. It should load all the parameters
    // from a configuration file
    sim.init();   

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

    // Run rollout
    sim.run(100); // Run for 100 steps
   
    return 0;
}