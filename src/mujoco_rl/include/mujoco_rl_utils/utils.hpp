#include <mujoco/mujoco.h>
#include <memory>
/*
    There was the need to create those destructors so that I can use smart pointers
 */
struct MjModelDeleter {
    void operator()(mjModel* ptr) const {
        if (ptr) mj_deleteModel(ptr);
    }
};

struct MjDataDeleter {
    void operator()(mjData* ptr) const {
        if (ptr) mj_deleteData(ptr);
    }
};

using MjModelPtr = std::unique_ptr<mjModel, MjModelDeleter>;
using MjDataPtr = std::unique_ptr<mjData, MjDataDeleter>;