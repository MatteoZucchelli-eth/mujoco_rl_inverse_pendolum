#include <iostream>
#include <mujoco_rl/sim.hpp>

int main() {
    std::cout << "Creating the Sim" << std::endl;

    Sim sim;
    sim.create_model("/workspaces/magic-triangle/src/model/scene.xml");
    sim.init();
    sim.step_parallel();
    
    return 0;
}