#include <rl_controller/controller.hpp>
#include <filesystem>
#include <iostream>

namespace rl {

Controller::Controller(float* action_buf, float* obs_buf,
                       float* log_prob_buf, float* value_buf,
                       int num_envs, int n_cores)
    : action_buf_(action_buf),
      obs_buf_(obs_buf),
      log_prob_buf_(log_prob_buf),
      value_buf_(value_buf),
      num_envs_(num_envs)
{
    envs_per_thread_ = (num_envs + n_cores - 1) / n_cores;
}

Controller::~Controller() {}

void Controller::init(float action_lo, float action_hi)
{
    network_   = ActorCritic(obs_dim, action_dim, action_lo, action_hi);
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(),
        torch::optim::AdamOptions(3e-4));

    network_->eval();  // start in eval mode for rollout collection
}

// ---------------------------------------------------------------------------
// computeActions — called once per rollout step, possibly from multiple threads
// ---------------------------------------------------------------------------
void Controller::computeActions(int thread_id)
{
    int start = thread_id * envs_per_thread_;
    int end   = start + envs_per_thread_;

    for (int i = start; i < end && i < num_envs_; ++i) {
        const float* obs_ptr  = obs_buf_      + i * obs_dim;
        float*       act_ptr  = action_buf_   + i * action_dim;
        float*       lp_ptr   = log_prob_buf_ + i;
        float*       val_ptr  = value_buf_    + i;

        ActorCriticOutput out = network_->infer(obs_ptr);

        for (int k = 0; k < action_dim; ++k)
            act_ptr[k] = (k < (int)out.actions.size()) ? out.actions[k] : 0.0f;

        *lp_ptr  = out.log_prob;
        *val_ptr = out.value;
    }
}

// ---------------------------------------------------------------------------
// updatePolicy — PPO joint actor-critic update
// ---------------------------------------------------------------------------
TrainingMetrics Controller::updatePolicy(
    const std::vector<float>& observations,
    const std::vector<float>& actions,
    const std::vector<float>& log_probs_old,
    const std::vector<float>& returns,
    const std::vector<float>& advantages)
{
    int64_t N = (int64_t)observations.size() / obs_dim;

    // Build tensors (clone so training cannot corrupt the rollout buffers)
    auto obs_t     = torch::from_blob((void*)observations.data(), {N, obs_dim},    torch::kFloat32).clone();
    auto act_t     = torch::from_blob((void*)actions.data(),      {N, action_dim}, torch::kFloat32).clone();
    auto lp_old_t  = torch::from_blob((void*)log_probs_old.data(),{N, 1},          torch::kFloat32).clone();
    auto ret_t     = torch::from_blob((void*)returns.data(),      {N, 1},          torch::kFloat32).clone();
    auto adv_t     = torch::from_blob((void*)advantages.data(),   {N, 1},          torch::kFloat32).clone();

    // Normalize advantages over the whole rollout batch
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8f);

    // PPO hyperparameters
    const int   epochs      = 10;
    const int64_t batch_size = 512;
    const float clip_param  = 0.2f;
    const float entropy_coef = 0.002f;
    const float value_coef   = 0.5f;
    const float max_grad_norm = 0.5f;

    network_->train();

    double sum_actor_loss  = 0.0;
    double sum_critic_loss = 0.0;
    double sum_entropy     = 0.0;
    int    num_updates     = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        auto indices = torch::randperm(N, torch::kLong);

        for (int64_t i = 0; i < N; i += batch_size) {
            auto idx = indices.slice(0, i, std::min(i + batch_size, N));

            auto obs_b    = obs_t.index_select(0, idx);
            auto act_b    = act_t.index_select(0, idx);
            auto lp_old_b = lp_old_t.index_select(0, idx);
            auto ret_b    = ret_t.index_select(0, idx);
            auto adv_b    = adv_t.index_select(0, idx);

            // Forward pass through the unified network
            auto [log_probs, entropy, values] = network_->evaluate(obs_b, act_b);

            // --- Critic loss ---
            auto critic_loss = torch::mse_loss(values, ret_b);

            // --- Actor loss (PPO clipped surrogate) ---
            auto ratio  = torch::exp(log_probs - lp_old_b);
            auto surr1  = ratio * adv_b;
            auto surr2  = torch::clamp(ratio, 1.0f - clip_param, 1.0f + clip_param) * adv_b;
            auto actor_loss = -torch::min(surr1, surr2).mean();

            // --- Entropy bonus (maximise entropy: subtract from total loss) ---
            auto entropy_loss = -entropy.mean();

            // --- Combined loss ---
            auto total_loss = actor_loss
                            + value_coef   * critic_loss
                            + entropy_coef * entropy_loss;

            optimizer_->zero_grad();
            total_loss.backward();
            torch::nn::utils::clip_grad_norm_(network_->parameters(), max_grad_norm);
            optimizer_->step();

            sum_actor_loss  += actor_loss.item<double>();
            sum_critic_loss += critic_loss.item<double>();
            sum_entropy     += entropy.mean().item<double>();
            ++num_updates;
        }
    }

    network_->eval();  // back to eval mode for next rollout

    TrainingMetrics m;
    if (num_updates > 0) {
        m.actor_loss  = sum_actor_loss  / num_updates;
        m.critic_loss = sum_critic_loss / num_updates;
        m.entropy     = sum_entropy     / num_updates;
    }

    std::cout << "  Actor Loss: "  << m.actor_loss
              << " | Critic Loss: " << m.critic_loss
              << " | Entropy: "     << m.entropy << std::endl;
    return m;
}

// ---------------------------------------------------------------------------
// Checkpoint helpers
// ---------------------------------------------------------------------------
void Controller::save(const std::string& directory, int iteration)
{
    std::string path = directory + "/network_" + std::to_string(iteration) + ".pt";
    torch::save(network_, path);
    std::cout << "Saved model → " << path << std::endl;
}

void Controller::load(const std::string& network_path)
{
    torch::load(network_, network_path);
    network_->eval();
    std::cout << "Loaded model ← " << network_path << std::endl;
}

} // namespace rl

