#pragma once

#include <mujoco/mujoco.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <mujoco_rl_utils/utils.hpp>
#include <rl_controller/controller.hpp>
#include <cstdlib>

// Forward decl
namespace rl { class Controller; }

namespace mj_pool {

class Sim {
public:
    Sim();
    void init();
    void create_model(const char* filename);
    void create_data();
    void set_controller(std::shared_ptr<rl::Controller> controller);
    void run(int steps);

    float* get_observation_buffer(int thread_id);
    float* get_action_buffer(int thread_id);
    float* get_log_prob_buffer(int thread_id);
    float* get_value_buffer(int thread_id);
    float* get_reward_buffer(int thread_id);
    
    float* get_observation_buffer();
    float* get_action_buffer();
    float* get_log_prob_buffer();
    float* get_value_buffer();

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
    const double gamma = 0.9;

    int state_dim_;

    int n_cores_;
    int envs_per_thread_;

    // Current Step Buffers
    std::vector<float> global_observation_buffer;
    std::vector<float> global_action_buffer;
    std::vector<float> global_log_prob_buffer;
    std::vector<float> global_value_buffer;
    std::vector<float> global_reward_buffer;

    // Rollout Buffers (History)
    std::vector<float> rollout_observations;
    std::vector<float> rollout_actions;
    std::vector<float> rollout_log_probs;
    std::vector<float> rollout_values;
    std::vector<float> rollout_rewards;
    std::vector<float> rollout_returns; // Store returns (rewards to go)
    std::vector<float> rollout_advantages;
    std::vector<bool> rollout_dones;

    std::vector<double> global_simstate_buffer;
    std::vector<double> global_initial_state_buffer;
    std::vector<bool> global_done_buffer;

    std::shared_ptr<rl::Controller> controller_;

    void worker_thread(int thread_id, int envs_per_thread);

    void apply_actions_thread(int thread_id);
    void get_observations_thread(int thread_id);
    void run_physics_thread(int thread_id);

    void load_state_from_buffer(int env_id, mjData* d, const std::vector<double>& buffer);
    void save_state_to_buffer(int env_id, mjData* d, std::vector<double>& buffer);

    void serialize_state(const mjData* d, double* dst);
    void deserialize_state(mjData* d, const double* src);

    void add_noise(mjData* d);
    double compute_reward(const mjData* d);
    void store_rollout_step(int step_idx, int env_id);
    void step_parallel(int step_idx);
    void compute_returns(int steps);
    void compute_advantages(int steps);
    void train();
};
}