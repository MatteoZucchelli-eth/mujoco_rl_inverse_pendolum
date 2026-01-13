#include <iostream>
#include <mujoco_rl/sim.hpp>

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

    while (i < 10000) {
        sim.step_parallel();
        i++;
    }
    
    return 0;
}