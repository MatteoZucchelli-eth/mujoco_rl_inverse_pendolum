#include <mujoco/mujoco.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <mujoco_rl_utils/utils.hpp>

class Sim {
public:
    Sim();
    void init();
    void step_parallel();
    void create_model(const char* filename);
    void create_data();

private:
    MjModelPtr m_;
    std::vector<MjDataPtr> d_;

    // Settings
    int num_envs = 1000;
    int obs_dim = 10;
    int action_dim = 10;
    double max_sim_time_ = 10;

    int state_dim_;

    int n_cores_;
    int envs_per_thread_;

    std::vector<float> global_observation_buffer;
    std::vector<float> global_action_buffer;
    std::vector<double> global_simstate_buffer;
    std::vector<bool> global_done_buffer;

    void worker_thread(int thread_id, int envs_per_thread);

    void save_simstate(int env_id, mjData* d);
    void load_simstate(int env_id, mjData* d);

    void apply_actions_thread(int thread_id);
    void get_observations_thread(int thread_id);
    void run_physics_thread(int thread_id);
};