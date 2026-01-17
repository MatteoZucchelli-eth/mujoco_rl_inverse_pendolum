#include <mujoco/mujoco.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <mujoco_rl_utils/utils.hpp>

namespace mj_pool {

class Sim {
public:
    Sim();
    void init();
    void step_parallel();
    void create_model(const char* filename);
    void create_data();

    float* get_observation_buffer();
    float* get_action_buffer();

private:
    MjModelPtr m_;
    std::vector<MjDataPtr> d_;

    // Settings
    const int num_envs = 1000;
    const int obs_dim = 8;
    const int action_dim = 1;
    const double max_sim_time_ = 10;
    const double noise_min = -0.01;
    const double noise_max = 0.01;

    int state_dim_;

    int n_cores_;
    int envs_per_thread_;

    std::vector<float> global_observation_buffer;
    std::vector<float> global_action_buffer;
    std::vector<double> global_simstate_buffer;
    std::vector<double> global_initial_state_buffer;
    std::vector<bool> global_done_buffer;

    void worker_thread(int thread_id, int envs_per_thread);

    void apply_actions_thread(int thread_id);
    void get_observations_thread(int thread_id);
    void run_physics_thread(int thread_id);

    void load_state_from_buffer(int env_id, mjData* d, const std::vector<double>& buffer);
    void save_state_to_buffer(int env_id, mjData* d, std::vector<double>& buffer);

    void serialize_state(const mjData* d, double* dst);
    void deserialize_state(mjData* d, const double* src);

    void add_noise(mjData* d);
};

}