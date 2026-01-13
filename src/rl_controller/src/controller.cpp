#include <rl_controller/controller.hpp>

namespace rl {
    Controller::Controller(std::vector<float> &global_action_buffer) : globalActionBuffer(global_action_buffer) {}

    // std::vector<float> Controller::computeActions(const std::vector<float> &observations) {
    //     // Preparation of the input + Infrence of the model 

    //     // model_.forward(observations);

    //     // actions = Controller::outputToActions(output);
    //     return actions;
    // }
    
    Controller::~Controller() {}
}
