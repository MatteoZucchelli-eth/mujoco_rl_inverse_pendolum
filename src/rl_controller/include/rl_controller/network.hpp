#pragma once

#include <torch/torch.h>
#include <vector>

namespace rl {

// Returned by infer() - single-environment inference (no grad)
struct ActorCriticOutput {
    std::vector<float> actions;  // scaled to [action_lo, action_hi]
    float log_prob;
    float value;
};

// Unified Actor-Critic with shared trunk.
//
// Architecture:
//   LayerNorm(obs_dim)
//   → Linear(obs_dim, 256) + Tanh
//   → Linear(256, 256)     + Tanh
//   → Linear(256, 256)     + Tanh
//   actor head : Linear(256, 2*action_dim)  — mean + log_std per dimension
//   critic head: Linear(256, 1)             — state value V(s)
//
// Actions are squashed with tanh then linearly mapped to [action_lo, action_hi].
struct ActorCriticImpl : torch::nn::Module {
    ActorCriticImpl(int obs_dim, int action_dim, float action_lo, float action_hi);

    // Single-environment inference (no gradients).
    ActorCriticOutput infer(const float* obs_ptr);

    // Batch evaluation used during training.
    // Returns (log_probs [N,1], entropy [N,1], values [N,1]).
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
    evaluate(const torch::Tensor& obs, const torch::Tensor& actions);

private:
    // Shared trunk: returns (actor_raw [B, 2*action_dim], value [B, 1])
    std::pair<torch::Tensor, torch::Tensor> trunk_forward(const torch::Tensor& obs);

    torch::nn::LayerNorm norm_{nullptr};
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
    torch::nn::Linear actor_head_{nullptr};
    torch::nn::Linear critic_head_{nullptr};

    int obs_dim_;
    int action_dim_;
    float action_lo_, action_hi_;
    float action_scale_, action_bias_;  // scale=(hi-lo)/2,  bias=(hi+lo)/2
};

TORCH_MODULE(ActorCritic);

} // namespace rl

