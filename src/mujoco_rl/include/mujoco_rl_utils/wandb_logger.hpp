#pragma once

#include <cstdio>
#include <string>
#include <map>

// ---------------------------------------------------------------------------
// WandbLogger — header-only helper that streams metrics to a Python W&B daemon.
//
// Usage:
//   WandbLogger logger;
//   logger.init("my_project", "run_001");
//   logger.log(step, {{"reward/episode", 12.3}, {"train/actor_loss", 0.05}});
//   logger.finish();
// ---------------------------------------------------------------------------
class WandbLogger {
public:
    WandbLogger() = default;

    // Starts the Python wandb_daemon.py subprocess.
    // Returns true on success.
    bool init(const std::string& project,
              const std::string& run_name,
              const std::string& daemon_path =
                  "/workspaces/inverse_pendolum_training/scripts/wandb_daemon.py")
    {
        std::string cmd = "python3 -u \"" + daemon_path + "\""
                          " --project \"" + project  + "\""
                          " --name \""    + run_name + "\"";
        pipe_ = popen(cmd.c_str(), "w");
        initialized_ = (pipe_ != nullptr);
        if (!initialized_) {
            fprintf(stderr, "[WandbLogger] Failed to start daemon: %s\n",
                    cmd.c_str());
        }
        return initialized_;
    }

    bool is_initialized() const { return initialized_; }

    // Log a set of key-value metrics at the given step.
    void log(int step, const std::map<std::string, double>& metrics)
    {
        if (!initialized_ || !pipe_) return;

        // Build a simple inline JSON: {"step": N, "key": value, ...}
        std::string json = "{\"step\": " + std::to_string(step);
        for (const auto& [key, val] : metrics) {
            json += ", \"" + key + "\": " + std::to_string(val);
        }
        json += "}\n";

        fputs(json.c_str(), pipe_);
        fflush(pipe_);
    }

    // Send a finish signal and close the pipe.
    void finish()
    {
        if (!pipe_) return;
        fputs("{\"__finish__\": true}\n", pipe_);
        fflush(pipe_);
        pclose(pipe_);
        pipe_        = nullptr;
        initialized_ = false;
    }

    ~WandbLogger() { finish(); }

    // Non-copyable, moveable
    WandbLogger(const WandbLogger&)            = delete;
    WandbLogger& operator=(const WandbLogger&) = delete;
    WandbLogger(WandbLogger&&)                 = default;
    WandbLogger& operator=(WandbLogger&&)      = default;

private:
    FILE* pipe_        = nullptr;
    bool  initialized_ = false;
};
