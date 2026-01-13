#include <stdio.h>
#include <vector>

namespace rl {
    class Controller {
        public:
            Controller(std::vector<float> &global_action_buffer);
            ~Controller();
            std::vector<float> computeActions(const std::vector<float> &observations);
        private:
            std::vector<float> &globalActionBuffer;
    };
}