#include <iostream>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>

int main(int argc, char** argv) {
    std::cout << "Creating the Sim" << std::endl;

    std::string model_path = "/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml";
    if (argc > 1) {
        model_path = argv[1];
    }
    int i = 0;
    Sim sim;
    sim.create_model(model_path.c_str());
    sim.init();
    // rl::rlController controller(obs buffer from simulation basically so that it knows where to modify it. Think about a proper pipe line to let the controller know where to modify stuff);

    while (i < 10) {
        sim.step_parallel();
        i++;
    }
    
    return 0;
}