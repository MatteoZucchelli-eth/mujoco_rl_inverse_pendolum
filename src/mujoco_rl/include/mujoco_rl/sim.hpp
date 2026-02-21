#pragma once

#include <mujoco/mujoco.h>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <stdio.h>
#include <vector>
#include <omp.h>
#include <cmath>
#include <mujoco_rl_utils/utils.hpp>
#include <mujoco_rl_utils/wandb_logger.hpp>
#include <rl_controller/controller.hpp>

// Forward declarations
namespace rl { class Controller; struct TrainingMetrics; }

namespace mj_pool {

class Sim {
public:
    Sim();
    void init(const char* filename);
    void create_data();
    void set_controller(std::shared_ptr<rl::Controller> controller);

    // Optionally enable W&B logging before the first call to run().
    void enable_wandb(const std::string& project, const std::string& run_name);

    void run(int steps);

    // Raw buffer accessors used by Controller at construction time.
    float* get_observation_buffer();
    float* get_action_buffer();
    float* get_log_prob_buffer();
    float* get_value_buffer();
    float* get_reward_buffer();  // reward for all environments, indexed by env_id

    double get_accumulated_reward(int env_id);

    // --- Visualisation helpers ---
    mjModel* get_model() { return m_.get(); }
    int      get_num_envs() const { return num_envs; }

    // Perform one complete step (inference + physics with decimation).
    void step_inference();

    // Sync an external mjData with the internal state buffer.
    void load_state_to_mjdata(int env_id, mjData* d);

    // Externally override the state of one environment.
    void set_env_state(int env_id, const std::vector<double>& qpos,
                       const std::vector<double>& qvel);

private:
    MjModelPtr m_;
    std::vector<MjDataPtr> d_;

    // ---- Simulation settings ----
    const int    num_envs        = 104;
    const int    obs_dim         = 4;     // qpos(2) + qvel(2)
    const int    action_dim      = 1;
    const double max_sim_time_   = 20.0;
    const double noise_min       = -0.1;
    const double noise_max       =  0.1;
    const double gamma           = 0.99;
    const int    decimation      = 6;
    const double angle_threshold = 0.1;

    int state_dim_      = 0;
    int n_cores_        = 1;
    int envs_per_thread_= 1;

    // ---- Current-step buffers ----
    std::vector<float>  global_observation_buffer;
    std::vector<float>  global_action_buffer;
    std::vector<float>  global_log_prob_buffer;
    std::vector<float>  global_value_buffer;
    std::vector<float>  global_reward_buffer;

    // ---- Episode tracking ----
    std::vector<double> env_accumulated_reward;
    std::vector<double> env_accumulated_length;
    std::vector<double> completed_episode_rewards;
    std::vector<double> completed_episode_lengths;
    omp_lock_t          stats_lock;

    // ---- Rollout buffers (filled over one call to run()) ----
    std::vector<float> rollout_observations;
    std::vector<float> rollout_actions;
    std::vector<float> rollout_log_probs;
    std::vector<float> rollout_values;
    std::vector<float> rollout_rewards;
    std::vector<float> rollout_returns;
    std::vector<float> rollout_advantages;
    std::vector<bool>  rollout_dones;

    // ---- State persistence ----
    std::vector<double> global_simstate_buffer;
    std::vector<double> global_initial_state_buffer;
    std::vector<bool>   global_done_buffer;

    std::shared_ptr<rl::Controller> controller_;

    // ---- W&B ----
    WandbLogger wandb_logger_;
    int         current_iteration_ = 0;

    // ---- Private helpers ----
    void create_model(const char* filename);

    void serialize_state  (const mjData* d, double* dst);
    void deserialize_state(mjData* d, const double* src);

    void save_state_to_buffer  (int env_id, mjData* d,       std::vector<double>& buf);
    void load_state_from_buffer(int env_id, mjData* d, const std::vector<double>& buf);

    void add_noise(mjData* d);
    double compute_reward(const mjData* d);
    void store_rollout_step(int step_idx, int env_id);
    void step_parallel(int step_idx);
    void compute_gae(int steps);

    // Runs one PPO update and returns training metrics.
    rl::TrainingMetrics train();
};

} // namespace mj_pool
