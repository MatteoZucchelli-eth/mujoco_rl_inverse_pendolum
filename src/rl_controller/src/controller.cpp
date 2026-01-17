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
    
    Controller::~Controller() {}
}
