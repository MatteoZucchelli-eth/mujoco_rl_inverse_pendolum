#include <rl_controller/network.hpp>
#include <cmath>

namespace rl {

// ---------------------------------------------------------------------------
// ActorCriticImpl
// ---------------------------------------------------------------------------

ActorCriticImpl::ActorCriticImpl(int obs_dim, int action_dim,
                                 float action_lo, float action_hi)
    : obs_dim_(obs_dim),
      action_dim_(action_dim),
      action_lo_(action_lo),
      action_hi_(action_hi),
      action_scale_((action_hi - action_lo) / 2.0f),
      action_bias_((action_hi + action_lo) / 2.0f)
{
    norm_        = register_module("norm",        torch::nn::LayerNorm(torch::nn::LayerNormOptions({obs_dim})));
    fc1_         = register_module("fc1",         torch::nn::Linear(obs_dim, 256));
    fc2_         = register_module("fc2",         torch::nn::Linear(256, 256));
    fc3_         = register_module("fc3",         torch::nn::Linear(256, 256));
    actor_head_  = register_module("actor_head",  torch::nn::Linear(256, 2 * action_dim));
    critic_head_ = register_module("critic_head", torch::nn::Linear(256, 1));
}

std::pair<torch::Tensor, torch::Tensor>
ActorCriticImpl::trunk_forward(const torch::Tensor& obs)
{
    auto x = norm_(obs);
    x = torch::tanh(fc1_->forward(x));
    x = torch::tanh(fc2_->forward(x));
    x = torch::tanh(fc3_->forward(x));
    return {actor_head_->forward(x), critic_head_->forward(x)};
}

ActorCriticOutput ActorCriticImpl::infer(const float* obs_ptr)
{
    torch::NoGradGuard no_grad;

    auto input = torch::from_blob(
        const_cast<float*>(obs_ptr), {1, obs_dim_},
        torch::TensorOptions().dtype(torch::kFloat32)).clone();

    auto [actor_raw, value_t] = trunk_forward(input);

    // Split into mean and log_std; clamp for numerical stability
    auto chunks  = actor_raw.chunk(2, /*dim=*/1);
    auto mu      = chunks[0];
    auto log_std = torch::clamp(chunks[1], -2.0f, 2.0f);
    auto std_dev = torch::exp(log_std);

    // Sample gaussian and squash
    auto u        = mu + torch::randn_like(mu) * std_dev;
    auto tanh_u   = torch::tanh(u);

    // Log-prob with tanh correction and action-scale Jacobian
    // log π(a|s) = log N(u; μ, σ) − log|da/du|
    // da/du = action_scale * (1 − tanh²(u))
    static const float log2pi_half = 0.5f * std::log(2.0f * M_PI);
    auto log_prob_gaussian = -0.5f * torch::pow((u - mu) / std_dev, 2) - log_std
                             - log2pi_half;
    auto log_det = torch::log(action_scale_ * (1.0f - torch::pow(tanh_u, 2) + 1e-6f));
    auto log_prob = (log_prob_gaussian - log_det).sum(/*dim=*/1);

    // Scale to [action_lo, action_hi]
    auto scaled = tanh_u * action_scale_ + action_bias_;
    scaled = scaled.contiguous();

    std::vector<float> actions(scaled.data_ptr<float>(),
                               scaled.data_ptr<float>() + scaled.numel());
    return {actions, log_prob.item<float>(), value_t.item<float>()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ActorCriticImpl::evaluate(const torch::Tensor& obs, const torch::Tensor& actions)
{
    // obs    : [N, obs_dim]
    // actions: [N, action_dim]  (already scaled to [action_lo, action_hi])

    auto [actor_raw, values] = trunk_forward(obs);

    auto chunks  = actor_raw.chunk(2, /*dim=*/1);
    auto mu      = chunks[0];
    auto log_std = torch::clamp(chunks[1], -2.0f, 2.0f);
    auto std_dev = torch::exp(log_std);

    // Invert scaling to get tanh output, then invert tanh to get u
    auto tanh_u = torch::clamp((actions - action_bias_) / action_scale_,
                               -0.999999f, 0.999999f);
    auto u      = torch::atanh(tanh_u);

    // Log-prob
    static const float log2pi_half = 0.5f * std::log(2.0f * M_PI);
    auto log_prob_gaussian = -0.5f * torch::pow((u - mu) / std_dev, 2) - log_std
                             - log2pi_half;
    auto log_det  = torch::log(action_scale_ * (1.0f - torch::pow(tanh_u, 2) + 1e-6f));
    auto log_probs = (log_prob_gaussian - log_det).sum(/*dim=*/1, /*keepdim=*/true);

    // Gaussian entropy: H = 0.5*(1 + log(2π)) + log_std  (per dimension, summed)
    auto entropy = (0.5f + 0.5f * std::log(2.0f * M_PI) + log_std)
                       .sum(/*dim=*/1, /*keepdim=*/true);

    return {log_probs, entropy, values};
}

} // namespace rl

