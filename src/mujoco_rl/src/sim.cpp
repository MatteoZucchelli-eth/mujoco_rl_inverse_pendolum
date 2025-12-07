#include <mujoco_rl/sim.hpp>

Sim::Sim() {

}

void Sim::create_model(std::string model_path) {
    mjModel* m = mj_loadXML(model_path, NULL, errstr, errstr_sz);
}