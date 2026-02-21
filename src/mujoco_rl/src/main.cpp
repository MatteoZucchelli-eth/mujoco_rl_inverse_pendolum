#include <iostream>
#include <filesystem>
#include <string>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>

namespace fs = std::filesystem;

static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " [options]\n\n"
              << "Options:\n"
              << "  --checkpoint <path>   Resume training from a saved network_N.pt file\n"
              << "  --project    <name>   Weights & Biases project name (enables W&B logging)\n"
              << "  --run        <name>   Weights & Biases run name (optional, defaults to auto)\n"
              << "\nExamples:\n"
              << "  " << prog << "                                      # fresh start, no W&B\n"
              << "  " << prog << " --project my-project                 # fresh start with W&B\n"
              << "  " << prog << " --checkpoint checkpoints/network_100.pt --project my-project\n";
}

int main(int argc, char** argv) {

    // ---------------------------------------------------------------------------
    // Argument parsing  (--flag value style, all optional)
    // ---------------------------------------------------------------------------
    std::string checkpoint_path;
    std::string wandb_project;
    std::string wandb_run_name;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--checkpoint" || arg == "-c") && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if ((arg == "--project" || arg == "-p") && i + 1 < argc) {
            wandb_project = argv[++i];
        } else if ((arg == "--run" || arg == "-r") && i + 1 < argc) {
            wandb_run_name = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    // ---------------------------------------------------------------------------
    // Paths
    // ---------------------------------------------------------------------------
    const std::string checkpoint_dir = "/workspaces/inverse_pendolum_training/checkpoints_7";
    const std::string model_path     = "/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml";

    if (!fs::exists(checkpoint_dir))
        fs::create_directories(checkpoint_dir);

    // ---------------------------------------------------------------------------
    // Simulation
    // ---------------------------------------------------------------------------
    std::cout << "Creating the Sim" << std::endl;
    mj_pool::Sim sim;
    sim.init(model_path.c_str());

    // Read ctrlrange from the loaded MuJoCo model.
    mjModel* m = sim.get_model();
    float action_lo = (m->nu > 0) ? (float)m->actuator_ctrlrange[0] : -1.0f;
    float action_hi = (m->nu > 0) ? (float)m->actuator_ctrlrange[1] :  1.0f;
    std::cout << "Actuator control range: [" << action_lo << ", " << action_hi << "]" << std::endl;

    // ---------------------------------------------------------------------------
    // Controller
    // ---------------------------------------------------------------------------
    const int num_envs = 104;
    const int n_cores  = omp_get_num_procs();

    auto controller = std::make_shared<rl::Controller>(
        sim.get_action_buffer(),
        sim.get_observation_buffer(),
        sim.get_log_prob_buffer(),
        sim.get_value_buffer(),
        num_envs,
        n_cores
    );
    controller->init(action_lo, action_hi);

    // ---------------------------------------------------------------------------
    // Optional: resume from checkpoint
    // ---------------------------------------------------------------------------
    int start_iteration = 0;
    if (!checkpoint_path.empty()) {
        if (!fs::exists(checkpoint_path)) {
            std::cerr << "Error: checkpoint file not found: " << checkpoint_path << std::endl;
            return 1;
        }
        std::cout << "Loading checkpoint: " << checkpoint_path << std::endl;
        controller->load(checkpoint_path);

        // Parse iteration number from filename (network_N.pt or actor_N.pt)
        try {
            for (const char* prefix : {"network_", "actor_"}) {
                size_t pos = checkpoint_path.find(prefix);
                if (pos != std::string::npos) {
                    size_t start = pos + std::string(prefix).size();
                    size_t end   = checkpoint_path.find(".pt", start);
                    if (end != std::string::npos) {
                        start_iteration = std::stoi(checkpoint_path.substr(start, end - start));
                        std::cout << "Resuming from iteration " << start_iteration << std::endl;
                        break;
                    }
                }
            }
        } catch (...) {
            std::cout << "Could not parse iteration number from filename." << std::endl;
        }
    } else {
        std::cout << "Starting fresh (no checkpoint)." << std::endl;
    }

    // ---------------------------------------------------------------------------
    // Optional: Weights & Biases
    // ---------------------------------------------------------------------------
    if (!wandb_project.empty()) {
        sim.enable_wandb(wandb_project, wandb_run_name);
        std::cout << "W&B enabled — project: " << wandb_project << std::endl;
    } else {
        std::cout << "W&B disabled. Use --project <name> to enable." << std::endl;
    }

    sim.set_controller(controller);

    // ---------------------------------------------------------------------------
    // Training loop
    // ---------------------------------------------------------------------------
    const int num_iterations      = 200;
    const int steps_per_iteration = 3000;

    for (int i = start_iteration; i < start_iteration + num_iterations; ++i) {
        std::cout << "Iteration " << i + 1
                  << "/" << (start_iteration + num_iterations) << " ..." << std::endl;
        sim.run(steps_per_iteration);

        if ((i + 1) % 10 == 0)
            controller->save(checkpoint_dir, i + 1);
    }

    return 0;
}
