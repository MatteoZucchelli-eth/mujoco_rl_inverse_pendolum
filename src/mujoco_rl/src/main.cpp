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

    int i = 0;
    while (i < 10) {
        sim.step_parallel();
        i++;
    }


    
    return 0;
}