#include <rl_controller/controller.hpp>

namespace rl {
    Controller::Controller(float *global_action_buffer_ptr, float *global_observation_buffer_ptr, 
                           float *global_log_prob_buffer_ptr, float *global_value_buffer_ptr,
                           int num_envs, int n_cores) 
        : global_action_buffer_ptr_(global_action_buffer_ptr), 
          global_observation_buffer_ptr_(global_observation_buffer_ptr),
          global_log_prob_buffer_ptr_(global_log_prob_buffer_ptr),
          global_value_buffer_ptr_(global_value_buffer_ptr),
          num_envs_(num_envs) {
              envs_per_thread_ = (num_envs + n_cores - 1) / n_cores;
          }

    void Controller::init() {
        actor_ = Network::makeActor(obs_dim, action_dim);
        critique_ = Network::makeCritique(obs_dim);
        
        // Initialize optimizer
        critique_optimizer_ = std::make_unique<torch::optim::Adam>(critique_->parameters(), torch::optim::AdamOptions(1e-3));
        actor_optimizer_ = std::make_unique<torch::optim::Adam>(actor_->parameters(), torch::optim::AdamOptions(3e-4));
    }

    void Controller::updatePolicy(const std::vector<float>& observations, const std::vector<float>& actions, 
                                  const std::vector<float>& log_probs_old, const std::vector<float>& returns,
                                  const std::vector<float>& advantages) {
        
        int64_t num_samples = observations.size() / obs_dim;
        
        // Tensorize everything once
        auto obs_tensor = torch::from_blob((void*)observations.data(), {num_samples, obs_dim}, torch::kFloat32);
        auto act_tensor = torch::from_blob((void*)actions.data(), {num_samples, action_dim}, torch::kFloat32);
        auto log_probs_old_tensor = torch::from_blob((void*)log_probs_old.data(), {num_samples, 1}, torch::kFloat32);
        auto ret_tensor = torch::from_blob((void*)returns.data(), {num_samples, 1}, torch::kFloat32);
        auto adv_tensor = torch::from_blob((void*)advantages.data(), {num_samples, 1}, torch::kFloat32);

        // Normalize advantages
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8);

        // Hyperparameters
        int epochs = 10;
        int batch_size = 64;
        float clip_param = 0.2;

        actor_->train();
        critique_->train();

        auto dataset_size = num_samples;
        
        // Joint training loop (Interleaved updates)
        for (int epoch = 0; epoch < epochs; ++epoch) {
             auto indices = torch::randperm(dataset_size, torch::kLong);
             
             for (int64_t i = 0; i < dataset_size; i += batch_size) {
                 auto batch_indices = indices.slice(0, i, std::min(i + batch_size, dataset_size));
                 
                 auto obs_batch = obs_tensor.index_select(0, batch_indices);
                 auto act_batch = act_tensor.index_select(0, batch_indices);
                 auto old_log_prob_batch = log_probs_old_tensor.index_select(0, batch_indices);
                 auto ret_batch = ret_tensor.index_select(0, batch_indices);
                 auto adv_batch = adv_tensor.index_select(0, batch_indices);

                 // --- 1. Update Critique ---
                 critique_optimizer_->zero_grad();
                 auto predicted_values = critique_->forward(obs_batch);
                 auto value_loss = torch::mse_loss(predicted_values, ret_batch);
                 value_loss.backward();
                 critique_optimizer_->step();

                 // --- 2. Update Actor ---
                 actor_optimizer_->zero_grad();
                 auto new_log_prob = Network::evaluateActor(actor_, obs_batch, act_batch);
                 auto ratio = torch::exp(new_log_prob - old_log_prob_batch);
                 auto surr1 = ratio * adv_batch;
                 auto surr2 = torch::clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_batch;
                 auto actor_loss = -torch::min(surr1, surr2).mean();
                 actor_loss.backward();
                 actor_optimizer_->step();
             }
        }
    }

    void Controller::computeActions(int thread_id) {
        int start_index = thread_id * envs_per_thread_;
        int end_index = start_index + envs_per_thread_;

        for (int i = start_index; i < end_index; i++) {
            if (i >= num_envs_) break;

            // Compute pointers for this specific environment
            float* obs_ptr = global_observation_buffer_ptr_ + (i * obs_dim);
            float* action_ptr = global_action_buffer_ptr_ + (i * action_dim);
            float* log_prob_ptr = global_log_prob_buffer_ptr_ + i; // 1 float per env
            float* value_ptr = global_value_buffer_ptr_ + i;       // 1 float per env

            // Forward pass to get actions and log_probs
            ActorOutput output = Network::forwardActor(actor_, obs_ptr, obs_dim);
            
            // Forward pass to get value
            float value = Network::forwardCritique(critique_, obs_ptr, obs_dim);

            // Store actions in the global buffer
            for (int k = 0; k < action_dim; k++) {
                if (k < (int)output.action.size()) {
                    action_ptr[k] = output.action[k];
                }
            }
            
            // Store log_prob and value
            *log_prob_ptr = output.log_prob;
            *value_ptr = value;
        }
    }

    void Controller::save(const std::string& directory, int iteration) {
        std::string actor_path = directory + "/actor_" + std::to_string(iteration) + ".pt";
        std::string critique_path = directory + "/critique_" + std::to_string(iteration) + ".pt";
        
        torch::save(actor_, actor_path);
        torch::save(critique_, critique_path);
        std::cout << "Saved models at iteration " << iteration << std::endl; 
    }
    
    Controller::~Controller() {}
}
