#pragma once

#include <stdio.h>
#include <vector>
#include <string>
#include <torch/torch.h>
#include <rl_controller/network.hpp>

namespace rl {

struct TrainingMetrics {
    double actor_loss  = 0.0;
    double critic_loss = 0.0;
    double entropy     = 0.0;
};

class Controller {
public:
    Controller(float* action_buf, float* obs_buf,
               float* log_prob_buf, float* value_buf,
               int num_envs, int n_cores);
    ~Controller();

    // action_lo / action_hi are read from the MuJoCo model's ctrlrange.
    void init(float action_lo, float action_hi);

    // Compute actions for environments owned by thread_id.
    void computeActions(int thread_id);

    // Run one round of PPO updates. Returns training metrics.
    TrainingMetrics updatePolicy(const std::vector<float>& observations,
                                 const std::vector<float>& actions,
                                 const std::vector<float>& log_probs_old,
                                 const std::vector<float>& returns,
                                 const std::vector<float>& advantages);

    void save(const std::string& directory, int iteration);
    void load(const std::string& network_path);  // unified network_N.pt file

private:
    ActorCritic network_{nullptr};
    std::unique_ptr<torch::optim::Adam> optimizer_;

    float* action_buf_;
    float* obs_buf_;
    float* log_prob_buf_;
    float* value_buf_;

    const int obs_dim    = 4;
    const int action_dim = 1;

    int num_envs_;
    int envs_per_thread_;
};

} // namespace rl
