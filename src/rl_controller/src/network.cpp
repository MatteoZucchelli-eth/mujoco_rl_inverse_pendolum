#include <rl_controller/network.hpp>

namespace rl {

    torch::nn::Sequential Network::makeActor(int obs_dim, int action_dim) {
        torch::nn::Sequential model;
        // Input: obs_dim, Output: action_dim
        model->push_back(torch::nn::Linear(obs_dim, 64));
        model->push_back(torch::nn::Tanh());
        model->push_back(torch::nn::Linear(64, 32));
        model->push_back(torch::nn::Tanh());
        // Output mean and log_std (2 * action_dim)
        model->push_back(torch::nn::Linear(32, 2 * action_dim));
        return model;
    }

    torch::nn::Sequential Network::makeCritique(int obs_dim) {
        torch::nn::Sequential model;
        // Input: obs_dim, Output: Value (1)
        model->push_back(torch::nn::Linear(obs_dim, 64));
        model->push_back(torch::nn::ReLU());
        model->push_back(torch::nn::Linear(64, 32));
        model->push_back(torch::nn::ReLU());
        model->push_back(torch::nn::Linear(32, 1));
        return model;
    }

    ActorOutput Network::forwardActor(torch::nn::Sequential module, const float* obs_ptr, int obs_dim) {
        torch::NoGradGuard no_grad; // No gradients needed for inference
        // Create tensor from raw pointer
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob((void*)obs_ptr, {1, obs_dim}, options);

        torch::Tensor output = module->forward(input);

        // Output has shape [1, 2 * action_dim] -> Split into mean and log_std
        auto chunks = output.chunk(2, 1);
        auto mu = chunks[0];
        auto log_std = chunks[1];
        
        // Clamp log_std to maintain numerical stability
        log_std = torch::clamp(log_std, -20.0, 2.0);
        auto std_dev = torch::exp(log_std);

        // Sample from Gaussian
        auto epsilon = torch::randn_like(mu);
        auto action_unc = mu + epsilon * std_dev; // Unconstrained action
        
        // Squash to [-1, 1]
        auto action = torch::tanh(action_unc);

        // Calculate log probability
        // log_prob = log_prob_gaussian - log_det_jacobian_tanh
        // log_prob_gaussian = -0.5 * ((x - mu)/sigma)^2 - log(sigma) - 0.5 * log(2pi)
        // simplified using torch distributions or manually:
        // Here we do it manually for the sampled value
        auto log_prob = -0.5 * torch::pow((action_unc - mu) / std_dev, 2) - log_std - 0.5 * std::log(2 * M_PI);
        // Tanh correction: log(1 - tanh^2(x))
        // We sum over action dimensions (presumably action_dim=1 here so sum is trivial)
        log_prob = log_prob - torch::log(1.0 - torch::pow(action, 2) + 1e-6);
        log_prob = log_prob.sum(1); // Sum across action dimensions

        // Convert output tensor back to vector<float>
        action = action.contiguous();
        std::vector<float> actions(action.data_ptr<float>(), action.data_ptr<float>() + action.numel());
        
        return {actions, log_prob.item<float>()};
    }

    float Network::forwardCritique(torch::nn::Sequential module, const float* obs_ptr, int obs_dim) {
        torch::NoGradGuard no_grad;
        auto options = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor input = torch::from_blob((void*)obs_ptr, {1, obs_dim}, options);

        torch::Tensor output = module->forward(input);

        // Return scalar value
        return output.item<float>();
    }

}
