#pragma once

#include <torch/torch.h>

namespace rl {
    struct ActorOutput {
        std::vector<float> action;
        float log_prob;
    };

    class Network {
        public:
            static torch::nn::Sequential makeActor(int obs_dim, int action_dim);
            static torch::nn::Sequential makeCritique(int obs_dim);

            static ActorOutput forwardActor(torch::nn::Sequential module, const float* obs_ptr, int obs_dim);
            static float forwardCritique(torch::nn::Sequential module, const float* obs_ptr, int obs_dim);
    };
}
