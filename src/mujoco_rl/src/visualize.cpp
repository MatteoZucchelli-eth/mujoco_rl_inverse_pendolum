#include <iostream>
#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <mujoco_rl/sim.hpp>
#include <rl_controller/controller.hpp>
#include <thread>
#include <chrono>
#include <filesystem>
#include <mutex>
#include <atomic>

namespace fs = std::filesystem;

// Globals for callbacks
mjModel* m = nullptr;
mjData* d = nullptr; // Visualization data
mj_pool::Sim* g_sim = nullptr; // Global sim pointer
std::mutex mtx;
std::atomic<bool> run_sim(true);
std::atomic<float> last_action(0.0f); // Store last action for display
std::atomic<float> last_reward(0.0f); // Store last reward for display
std::atomic<float> cumulative_reward(0.0f); // Store cumulative reward for display

mjvCamera cam;
mjvOption opt;
mjvScene scn;
mjrContext con;

bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (!g_sim || !m || !d) return;

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        
        std::lock_guard<std::mutex> lock(mtx);

        if (key == GLFW_KEY_R) { // Reset to zero
            std::cout << "[User] Resetting environment..." << std::endl;
            std::vector<double> qpos(m->nq, 0.0);
            std::vector<double> qvel(m->nv, 0.0);
            
            // Set pendulum angle to PI
            if (m->nq > 1) {
                qpos[1] = M_PI; 
            } else if (m->nq == 1) {
                qpos[0] = M_PI;
            }

            g_sim->set_env_state(0, qpos, qvel);
        } else if (key == GLFW_KEY_RIGHT) { // Push
            std::cout << "[User] Push Right..." << std::endl;
            std::vector<double> qpos(m->nq);
            std::vector<double> qvel(m->nv);
            
            // Get current internal state
            // Hacky: we use our visualization data as proxy for 'current state' 
            // because we sync it frequently.
            mju_copy(qpos.data(), d->qpos, m->nq);
            mju_copy(qvel.data(), d->qvel, m->nv);
             
            int target_idx = (m->nq > 1) ? 1 : 0;
            qpos[target_idx] += 0.5; 
            g_sim->set_env_state(0, qpos, qvel);
             
        } else if (key == GLFW_KEY_LEFT) { // Push
            std::cout << "[User] Push Left..." << std::endl;
            std::vector<double> qpos(m->nq);
            std::vector<double> qvel(m->nv);
            mju_copy(qpos.data(), d->qpos, m->nq);
            mju_copy(qvel.data(), d->qvel, m->nv);
             
            int target_idx = (m->nq > 1) ? 1 : 0;
            qpos[target_idx] -= 0.5; 
            g_sim->set_env_state(0, qpos, qvel);
        }
    }
}


void scroll(GLFWwindow* window, double xoffset, double yoffset) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05 * yoffset, &scn, &cam);
}

void mouse_button(GLFWwindow* window, int button, int act, int mods) {
    // update button state
    button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
    button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);
    glfwGetCursorPos(window, &lastx, &lasty);

    // Custom Interaction with CTRL
    if (act == GLFW_PRESS && g_sim && m && d) {
        bool mod_ctrl = (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS || 
                         glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS);
        
        // If CTRL is held...
        if (mod_ctrl) {
            std::lock_guard<std::mutex> lock(mtx);
            
            if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                // RESET
                std::cout << "[Mouse] Resetting..." << std::endl;
                std::vector<double> qpos(m->nq, 0.0);
                std::vector<double> qvel(m->nv, 0.0);
                if (m->nq > 1) qpos[1] = 0.1; 

                g_sim->set_env_state(0, qpos, qvel);
            }
            else if (button == GLFW_MOUSE_BUTTON_LEFT) {
                // PERTURB (Push)
                std::cout << "[Mouse] Pushing..." << std::endl;
                std::vector<double> qpos(m->nq);
                std::vector<double> qvel(m->nv);
                
                mju_copy(qpos.data(), d->qpos, m->nq);
                mju_copy(qvel.data(), d->qvel, m->nv);

                int target_idx = (m->nq > 1) ? 1 : 0;
                qpos[target_idx] += 0.5; 
                
                g_sim->set_env_state(0, qpos, qvel);
            }
        }
    }
}

void mouse_move(GLFWwindow* window, double xpos, double ypos) {
    if (!button_left && !button_middle && !button_right) return;

    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    int width, height;
    glfwGetWindowSize(window, &width, &height);

    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS || 
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

    mjtMouse action;
    if (button_right) action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if (button_left) action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else action = mjMOUSE_ZOOM;

    mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}

void sim_loop() {
    while (run_sim) {
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (g_sim) {
                g_sim->step_inference();
                
                // Capture action for display
                float* acts = g_sim->get_action_buffer();
                if (acts) last_action = acts[0];

                // Capture reward for display
                float* rewards = g_sim->get_reward_buffer();
                if (rewards) last_reward = rewards[0];

                // Capture cumulative reward for display
                cumulative_reward = (float)g_sim->get_accumulated_reward(0);
            }
        }
        // Simulation Step is 0.1s (decimation 20 * 0.005). 
        // We sleep for 100ms to run at 1x real-time speed.
        // Note: This results in 10 FPS visualization updates.
        std::this_thread::sleep_until(start + std::chrono::milliseconds(100));
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_actor_checkpoint>" << std::endl;
        return 1;
    }
    std::string actor_path = argv[1];

    if (!fs::exists(actor_path)) {
        std::cerr << "Checkpoint not found at: " << actor_path << std::endl;
        if (fs::exists("../" + actor_path)) {
            actor_path = "../" + actor_path;
            std::cout << "Found checkpoint at: " << actor_path << std::endl;
        } else {
             std::cerr << "Could not find checkpoint file." << std::endl;
             return 1;
        }
    }

    std::cout << "Initializing Visualization..." << std::endl;

    mj_pool::Sim sim;
    
    sim.init("/workspaces/inverse_pendolum_training/src/inverse_pendolum_model/scene.xml");

    int num_envs = sim.get_num_envs();
    int n_cores = omp_get_num_procs();

    // Read ctrlrange from the loaded model
    mjModel* vis_model = sim.get_model();
    float action_lo = (vis_model->nu > 0) ? (float)vis_model->actuator_ctrlrange[0] : -1.0f;
    float action_hi = (vis_model->nu > 0) ? (float)vis_model->actuator_ctrlrange[1] :  1.0f;

    auto controller = std::make_shared<rl::Controller>(
        sim.get_action_buffer(),
        sim.get_observation_buffer(),
        sim.get_log_prob_buffer(),
        sim.get_value_buffer(),
        num_envs,
        n_cores
    );
    controller->init(action_lo, action_hi);
    controller->load(actor_path);
    sim.set_controller(controller);

    std::cout << "[DEBUG] Controller set." << std::endl;
    
    // Set global pointer only when fully initialized
    g_sim = &sim;

    m = sim.get_model();
    if (!m) { std::cout << "Model is null!" << std::endl; return -1; }

    mjData* vis_d = mj_makeData(m);
    if (!vis_d) { std::cout << "mj_makeData failed!" << std::endl; return -1; }
    d = vis_d;

    std::cout << "[DEBUG] Initializing GLFW" << std::endl;
    if (!glfwInit()) {
        std::cerr << "Could not initialize GLFW" << std::endl;
        return 1;
    }

    GLFWwindow* window = glfwCreateWindow(1200, 900, "Mujoco RL Visualization", NULL, NULL);
    if (!window) {
        std::cerr << "Could not create GLFW window." << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);
    
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);
    std::cout << "[DEBUG] Context Made." << std::endl;

    // Callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    // Initial camera
    cam.type = mjCAMERA_FREE;
    cam.azimuth = 90;
    cam.elevation = -10;
    cam.distance = 4.0;
    cam.lookat[0] = 0.0; cam.lookat[1] = 0.0; cam.lookat[2] = 0.5;

    // Start Physics Thread
    std::thread physics_thread(sim_loop);
    
    while (!glfwWindowShouldClose(window)) {
        
        // Sync state from Sim
        {
            std::lock_guard<std::mutex> lock(mtx);
            sim.load_state_to_mjdata(0, vis_d);
        }

        // Forward Kinematics for Rendering (safe to do here as vis_d is local to this thread mostly)
        mj_forward(m, vis_d);

        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        mjv_updateScene(m, vis_d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // Overlays
        char title[100] = "Inverted Pendulum RL";
        char content[500];
        
        sprintf(content, "Time: %.2f\nPos: %.2f\nAngle: %.2f\nAction: %.2f\nReward: %.2f\nCum Reward: %.2f\nFPS: %d", 
                vis_d->time, vis_d->qpos[0], vis_d->qpos[1], last_action.load(), last_reward.load(), cumulative_reward.load(), 60);

        mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, title, content, &con);
        
        mjr_overlay(mjFONT_NORMAL, mjGRID_BOTTOMLEFT, viewport, "Control Info", 
                    "Right Mouse: Pan | Left Mouse: Rotate\n"
                    "CTRL+Left Click: Perturb | CTRL+Right Click: Reset\n"
                    "Arrows: Perturb | 'R': Reset", &con);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    
    run_sim = false;
    physics_thread.join();

    mjv_freeScene(&scn);
    mjr_freeContext(&con);
    mj_deleteData(vis_d);
    glfwTerminate();

    return 0;
}
