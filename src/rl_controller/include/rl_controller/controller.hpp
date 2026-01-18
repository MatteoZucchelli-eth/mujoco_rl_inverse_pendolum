#pragma once

#include <stdio.h>
#include <vector>
#include <torch/torch.h>
#include <rl_controller/network.hpp>

namespace rl {
    class Controller {
        public:
            Controller(float *global_action_buffer_ptr, float *global_observation_buffer_ptr, 
                       float *global_log_prob_buffer_ptr, float *global_value_buffer_ptr,
                       int num_envs, int n_cores);
            ~Controller();
            void init();
            void computeActions(int thread_id);
            void updatePolicy(const std::vector<float>& observations, const std::vector<float>& actions, 
                              const std::vector<float>& log_probs_old, const std::vector<float>& returns,
                              const std::vector<float>& advantages);
            void save(const std::string& directory, int iteration);
            void load(const std::string& actor_path);

        private:
            torch::nn::Sequential actor_{nullptr};
            torch::nn::Sequential critique_{nullptr};
            
            std::unique_ptr<torch::optim::Adam> critique_optimizer_;
            std::unique_ptr<torch::optim::Adam> actor_optimizer_;

            float *global_action_buffer_ptr_;
            float *global_observation_buffer_ptr_;
            float *global_log_prob_buffer_ptr_;
            float *global_value_buffer_ptr_;

            const int obs_dim = 4;
            const int action_dim = 1;
            
            int num_envs_;
            int envs_per_thread_;
    };
}