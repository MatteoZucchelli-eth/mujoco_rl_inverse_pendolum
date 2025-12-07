#include <mujoco/mujoco.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <mujoco_rl_utils/utils.hpp>

class Sim {
public:
    Sim();
    void create_model(char* filename);
    void create_data();
private:
    MjModelPtr m_;
    std::vector<MjDataPtr> d_;

    // Settings
    int num_envs = 1000;
    int obs_dim = 10;
    int action_dim = 10;

    std::vector<float> global_observation_buffer;
    std::vector<float> global_action_buffer;

};