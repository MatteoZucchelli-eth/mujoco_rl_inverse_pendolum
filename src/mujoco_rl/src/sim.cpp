#include <mujoco_rl/sim.hpp>

namespace mj_pool {

// Helper function to normalize angle to [-pi, pi]
inline double normalize_angle(double angle) {
    while (angle > M_PI)  angle -= 2 * M_PI;
    while (angle < -M_PI) angle += 2 * M_PI;
    return angle;
}

Sim::Sim() {
    std::cout << "Hello world"  << std::endl;
}

void Sim::init(const char* filename) {

    create_model(filename);
    if (d_.empty()) {
        create_data();
    }

    // Ensure the first data pointer exists
    if (!d_[0]) {
        d_[0] = MjDataPtr(mj_makeData(m_.get())); // Temp data
    }

    for (int i = 0; i < num_envs; ++i) {
        // Reset the temporary data before saving initial states
        mj_resetData(m_.get(), d_[0].get());
        
        // Set initial pendulum angle to PI
        if (m_->nq >= 2) {
            d_[0]->qpos[1] = M_PI;
        } else if (m_->nq == 1) {
            d_[0]->qpos[0] = M_PI;
        }

        // Save this pristine initial state (never overwritten)
        save_state_to_buffer(i, d_[0].get(), global_initial_state_buffer);
        
        // Also save to current state buffer so first step can begin
        save_state_to_buffer(i, d_[0].get(), global_simstate_buffer);

        // Populate initial Observation so the agent doesn't receive zeros on the first step
        int obs_offset = i * obs_dim;
        
        // Copy qpos
        int qpos_limit = std::min((int)m_->nq, obs_dim);
        for (int k = 0; k < qpos_limit; k++) {
            float val = (float)d_[0]->qpos[k];
            if ((m_->nq >= 2 && k == 1) || (m_->nq == 1 && k == 0)) {
                val = (float)normalize_angle(val);
            }
            global_observation_buffer[obs_offset + k] = val;
        }
        
        // Copy qvel
        int qvel_limit = std::min((int)m_->nv, obs_dim - qpos_limit);
        for (int k = 0; k < qvel_limit; k++) {
            global_observation_buffer[obs_offset + qpos_limit + k] = (float)d_[0]->qvel[k];
        }
    }

    std::cout << "Simulation initialization completed" << std::endl;
}

void Sim::create_model(const char* filename) {
    
    // For debugging
    constexpr int error_sz = 1024;
    char error[error_sz];
    mjModel* m_raw = mj_loadXML(filename, nullptr, error, error_sz);
    if (!m_raw) {
        throw std::runtime_error(std::string(error));
    }

    m_ = MjModelPtr(m_raw);
    std::cout << "--- Checking Model Masses ---" << std::endl;

    // YOU NEED THIS LOOP
    for (int i = 0; i < m_->nbody; i++) {
        // 1. Get the name
        const char* name = mj_id2name(m_.get(), mjOBJ_BODY, i);
        if (!name) name = "world/unnamed";

        // 2. Get the mass
        double mass = m_->body_mass[i];

        // 3. Print
        std::cout << "ID " << i << " [" << name << "]: " << mass << " kg" << std::endl;
    }
    m_->opt.timestep = 0.005;

    state_dim_ = 
    1 +                 // d->time
    m_->nq +            // d->qpos
    m_->nv +            // d->qvel
    m_->na +            // d->act
    (3 * m_->nmocap) +  // d->mocap_pos
    (4 * m_->nmocap) +  // d->mocap_quat
    m_->nuserdata +     // d->userdata
    m_->nv;             // d->qacc_warmstart

    std::cout << "Model created with state dimension: " << state_dim_ << std::endl;
}

void Sim::create_data() {
    if (!m_) {
        throw std::runtime_error("Model not created before create_data(). Call create_model() first.");
    }

    if (state_dim_ <= 0) {
        throw std::runtime_error("state_dim_ is not set or invalid. Ensure create_model() ran successfully.");
    }

    // Check if the model has enough dimensions for our observation definition (qpos + qvel)
    if (m_->nq + m_->nv < obs_dim) {
        throw std::runtime_error("Model state (nq=" + std::to_string(m_->nq) + ", nv=" + std::to_string(m_->nv) 
            + ") is smaller than obs_dim (" + std::to_string(obs_dim) + ")");
    }
    if (m_->nu < action_dim) {
        throw std::runtime_error("Model nu (" + std::to_string(m_->nu) + ") is smaller than action_dim (" + std::to_string(action_dim) + ")");
    }

    n_cores_ = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(n_cores_);

    d_.resize(n_cores_); // One data structure for core

    for (int i= 0; i < n_cores_; i++) {
        d_[i] = MjDataPtr(mj_makeData(m_.get())); // Returned the raw pointer
    }

    // Distribute environments across threads. Use ceiling so we cover all envs.
    envs_per_thread_ = (num_envs + n_cores_ - 1) / n_cores_;

    // Resize global buffers so accesses like &global_simstate_buffer[offset]
    // are valid and we avoid out-of-bounds memory access. This also initializes the vectors to be filled with 0.0
    global_observation_buffer.resize((size_t)num_envs * (size_t)obs_dim);
    global_action_buffer.resize((size_t)num_envs * (size_t)action_dim);
    global_log_prob_buffer.resize((size_t)num_envs); // 1 per env
    global_value_buffer.resize((size_t)num_envs);    // 1 per env
    global_reward_buffer.resize((size_t)num_envs);   // 1 per env

    global_simstate_buffer.resize((size_t)num_envs * (size_t)state_dim_);
    global_initial_state_buffer.resize((size_t)num_envs * (size_t)state_dim_);  // Separate buffer for pristine initial states
    global_done_buffer.resize((size_t)num_envs);

    // Resize episode tracking
    env_accumulated_reward.assign((size_t)num_envs, 0.0);
    env_accumulated_length.assign((size_t)num_envs, 0.0);
    omp_init_lock(&stats_lock);

    std::cout << "Created data for device with number of cores: " << n_cores_ <<std::endl;
    std::cout << "Each core  will execute: " << envs_per_thread_ << " simulations" << std::endl;
    
}

void Sim::serialize_state(const mjData* d, double* dst) {
    double* ptr = dst;

    // 1. Time
    *ptr = d->time; ptr++;

    // 2. QPOS
    mju_copy(ptr, d->qpos, m_->nq); ptr += m_->nq;

    // 3. QVEL
    mju_copy(ptr, d->qvel, m_->nv); ptr += m_->nv;

    // 4. ACT
    if (m_->na > 0) {
        mju_copy(ptr, d->act, m_->na); ptr += m_->na;
    }
    
    // 5. MOCAP
    if (m_->nmocap > 0) {
        mju_copy(ptr, d->mocap_pos, 3 * m_->nmocap); ptr += 3 * m_->nmocap;
        mju_copy(ptr, d->mocap_quat, 4 * m_->nmocap); ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(ptr, d->userdata, m_->nuserdata); ptr += m_->nuserdata;
    }

    // 7. Warm Start
    mju_copy(ptr, d->qacc_warmstart, m_->nv); ptr += m_->nv;
}

// PRIVATE HELPER: Reads raw memory and puts it into mjData
void Sim::deserialize_state(mjData* d, const double* src) {
    const double* ptr = src;

    // 1. Time
    d->time = *ptr; ptr++;

    // 2. QPOS
    mju_copy(d->qpos, ptr, m_->nq); ptr += m_->nq;

    // 3. QVEL
    mju_copy(d->qvel, ptr, m_->nv); ptr += m_->nv;

    // 4. ACT
    if (m_->na > 0) {
        mju_copy(d->act, ptr, m_->na); ptr += m_->na;
    }
    
    // 5. MOCAP
    if (m_->nmocap > 0) {
        mju_copy(d->mocap_pos, ptr, 3 * m_->nmocap); ptr += 3 * m_->nmocap;
        mju_copy(d->mocap_quat, ptr, 4 * m_->nmocap); ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(d->userdata, ptr, m_->nuserdata); ptr += m_->nuserdata;
    }

    // 7. Warm Start
    mju_copy(d->qacc_warmstart, ptr, m_->nv); ptr += m_->nv;
}

// Generic wrapper that handles the offset calculation
// Pass 'global_simstate_buffer' OR 'global_initial_state_buffer'
void Sim::save_state_to_buffer(int env_id, mjData* d, std::vector<double>& buffer) {
    size_t offset = (size_t)env_id * (size_t)state_dim_;
    serialize_state(d, &buffer[offset]);
}

// Generic wrapper for loading
void Sim::load_state_from_buffer(int env_id, mjData* d, const std::vector<double>& buffer) {
    size_t offset = (size_t)env_id * (size_t)state_dim_;
    deserialize_state(d, &buffer[offset]);
}

void Sim::add_noise(mjData* d) {
    
    // Add noise to each position
    for (int i = 0; i < m_->nq; i++) {
        // Generate random number in [0, 1] and scale to [noise_min, noise_max]
        double noise = noise_min + (static_cast<double>(rand()) / RAND_MAX) * (noise_max - noise_min);
        d->qpos[i] += noise;
    }
}
void Sim::step_parallel(int step_idx) {
    #pragma omp parallel num_threads(n_cores_)
    {
        int thread_id = omp_get_thread_num();
        if (thread_id >= n_cores_) {
            // Safety check
        } else {

            mjData* data = d_[thread_id].get();
            mjModel* model = m_.get();
            
            int start_index = thread_id * envs_per_thread_;
            int end_index = start_index + envs_per_thread_;

            // Loop thrugh the batch assigned to this thread
            for (int i = start_index; i < end_index; i++)  {
            // stop if we exceed  the total number of environments
            if (i >= num_envs) break;
            
            // load the state from the buffer
            load_state_from_buffer(i, data, global_simstate_buffer);

            // apply action
            int action_offset = i * action_dim;
            for (int k=0; k < action_dim; k++) {
                data->ctrl[k] = global_action_buffer[action_offset + k];
            }

            double accumulated_reward = 0.0;
            bool done = false;

            for (int j = 0; j < decimation; j++) {
                mj_step(model, data);

                // Compute reward
                double r = compute_reward(data);
                accumulated_reward += r;

                // Check if episode is done
                done = (data->time > max_sim_time_); 
                
                // Limit Base Position
                if (model->nq >= 1) {
                    if (std::abs(data->qpos[0]) > 2.4) {
                        done = true;
                        accumulated_reward -= 1000.0; // Terminal penalty
                    }
                }

                if (done) break;
            }

            global_reward_buffer[i] = (float)accumulated_reward;
            global_done_buffer[i] = done;  

            // Update Episode Stats
            env_accumulated_reward[i] += accumulated_reward;
            env_accumulated_length[i] += 1.0;

            // Save transition to rollout buffer (obs_t, act_t, rew_t, val_t, log_prob_t)
            store_rollout_step(step_idx, i);
            // Handle Reset or Continue
            if (done) {
                // Store stats
                omp_set_lock(&stats_lock);
                completed_episode_rewards.push_back(env_accumulated_reward[i]);
                completed_episode_lengths.push_back(env_accumulated_length[i]);
                omp_unset_lock(&stats_lock);

                // Reset stats
                env_accumulated_reward[i] = 0.0;
                env_accumulated_length[i] = 0.0;

                // Load the pristine initial state
                load_state_from_buffer(i, data, global_initial_state_buffer);

                // Add randomization to the initial state
                add_noise(data);

                // Forward kinematics to ensure consistency
                mj_forward(model, data);
            }

            // Save the current state (new initial state if reset, regular state otherwise)
            save_state_to_buffer(i, data, global_simstate_buffer);

            // Update observation buffer (next observation so the agent can act)
            // Obs = [qpos, qvel]
            int obs_offset = i * obs_dim;
            
            // Check bounds to be safe, though create_data validted nq+nv >= obs_dim
            int qpos_limit = std::min((int)model->nq, obs_dim);
            
            // Copy qpos
            for (int k = 0; k < qpos_limit; k++) {
                float val = (float)data->qpos[k];
                if ((model->nq >= 2 && k == 1) || (model->nq == 1 && k == 0)) {
                    val = (float)normalize_angle(val);
                }
                global_observation_buffer[obs_offset + k] = val;
            }
            
            // Copy qvel. We continue filling the buffer from where qpos left off.
            // Ensure we don't overflow obs_dim even if model has more states than we need.
            int qvel_limit = std::min((int)model->nv, obs_dim - qpos_limit);
            
            for (int k = 0; k < qvel_limit; k++) {
                global_observation_buffer[obs_offset + qpos_limit + k] = (float)data->qvel[k];
            }
        }
        }
    }
}

float* Sim::get_observation_buffer() { return &global_observation_buffer[0]; }
float* Sim::get_action_buffer()      { return &global_action_buffer[0]; }
float* Sim::get_log_prob_buffer()    { return &global_log_prob_buffer[0]; }
float* Sim::get_value_buffer()       { return &global_value_buffer[0]; }
float* Sim::get_reward_buffer()      { return &global_reward_buffer[0]; }

double Sim::get_accumulated_reward(int env_id) {
    if (env_id >= 0 && env_id < num_envs) {
        return env_accumulated_reward[env_id];
    }
    return 0.0;
}

void Sim::set_controller(std::shared_ptr<rl::Controller> controller) {
    controller_ = controller;
}

void Sim::run(int steps) {
    if (!controller_) {
        throw std::runtime_error("Controller not set. Call set_controller() before run().");
    }

    // Resize rollout buffers
    rollout_observations.resize((size_t)steps * (size_t)num_envs * (size_t)obs_dim);
    rollout_actions.resize((size_t)steps * (size_t)num_envs * (size_t)action_dim);
    rollout_log_probs.resize((size_t)steps * (size_t)num_envs);
    rollout_values.resize((size_t)steps * (size_t)num_envs);
    rollout_rewards.resize((size_t)steps * (size_t)num_envs);
    rollout_returns.resize((size_t)steps * (size_t)num_envs);
    rollout_advantages.resize((size_t)steps * (size_t)num_envs);
    rollout_dones.resize((size_t)steps * (size_t)num_envs);

    for (int s = 0; s < steps; ++s) {
        // Parallel inference
        #pragma omp parallel num_threads(n_cores_)
        {
            int thread_id = omp_get_thread_num();
            // Ensure thread_id is within bounds, just in case
            if (thread_id < n_cores_) {
                 controller_->computeActions(thread_id);
            }
        }

        // Parallel physics step and store data
        // We pass the current step index 's' to store data in the correct slot of rollout buffers
        step_parallel(s);
    } 

    // Compute value of the "next" state (the one we landed in after the last step)
    // This provides the bootstrap value V(s_T)
    #pragma omp parallel num_threads(n_cores_)
    {
        int thread_id = omp_get_thread_num();
        controller_->computeActions(thread_id);
    }
   
    // Compute gae after collecting all steps
    compute_gae(steps);

    // Calculate and log mean reward
    double total_reward = 0.0;
    #pragma omp parallel for reduction(+:total_reward) num_threads(n_cores_)
    for (size_t i = 0; i < rollout_rewards.size(); ++i) {
        total_reward += (double)rollout_rewards[i];
    }
    double mean_reward = total_reward / (double)rollout_rewards.size();
    
    // Average Episode Return
    double mean_ep_reward = 0.0;
    double mean_ep_len = 0.0;
    if (!completed_episode_rewards.empty()) {
        double sum_ep_rew = 0;
        double sum_ep_len = 0;
        for(auto v : completed_episode_rewards) sum_ep_rew += v;
        for(auto v : completed_episode_lengths) sum_ep_len += v;
        mean_ep_reward = sum_ep_rew / completed_episode_rewards.size();
        mean_ep_len = sum_ep_len / completed_episode_lengths.size();
        
        // Clear for next iteration
        completed_episode_rewards.clear();
        completed_episode_lengths.clear();
    }

    std::cout << "Mean Batch Reward (Per Step): " << mean_reward
              << " | Mean Episode Return: " << mean_ep_reward
              << " | Mean Ep Length: "      << mean_ep_len << std::endl;

    // Train Policy and Value Function
    rl::TrainingMetrics tm = train();

    // Log all metrics to W&B (if enabled)
    if (wandb_logger_.is_initialized()) {
        wandb_logger_.log(current_iteration_, {
            {"rollout/mean_step_reward",  mean_reward},
            {"rollout/mean_ep_return",    mean_ep_reward},
            {"rollout/mean_ep_length",    mean_ep_len},
            {"train/actor_loss",          tm.actor_loss},
            {"train/critic_loss",         tm.critic_loss},
            {"train/entropy",             tm.entropy},
        });
    }
    ++current_iteration_;
}

rl::TrainingMetrics Sim::train() {
    return controller_->updatePolicy(rollout_observations, rollout_actions,
                                     rollout_log_probs, rollout_returns, rollout_advantages);
}

void Sim::enable_wandb(const std::string& project, const std::string& run_name) {
    wandb_logger_.init(project, run_name);
}

// Reward based on angle (target = pi/2) and angular velocity
double Sim::compute_reward(const mjData* d) {
    double base_pos = 0.0;
    double angle = 0.0;
    double vel_angle = 0.0;
    double control = d->ctrl[0]; 


    // Check bounds and assign variables
    if (m_->nq >= 2) {
        base_pos = d->qpos[0];
        angle = normalize_angle(d->qpos[1]);
        vel_angle = d->qvel[1]; // Velocity of the joint
    } else {
        // Fallback or single joint
        angle = normalize_angle(d->qpos[0]);
        vel_angle = d->qvel[0];
    }
    
    double reward = 0.0;
    double vel_max = 20;
    
    if (std::abs(vel_angle) > vel_max) {
        reward += -10 * ((std::abs(vel_angle) - vel_max) * (std::abs(vel_angle) - vel_max));
    } else {
        if (std::abs(angle) > angle_threshold) {
        // Here just swing, penalize a little the velocity but not too much
        reward += 0.5 * (std::cos(angle) - 1);
        reward += -0.01 * (vel_angle * vel_angle);
        } else {
            // 1. Term to maintain upright posture
            // Minimizing angle^2
            reward += 0.5;
            reward += -0.01 * (angle * angle);
            reward += -0.01 * (vel_angle * vel_angle);
            if (std::abs(vel_angle) < 0.5) {
                reward += 2.0;
            }
        }
    }
   
    
    // // 3. Term to not reach the end of the base (Shaping)
    reward += -0.001 * (control * control);  
    // reward += -0.1 * (base_pos * base_pos);
    // if (std::abs(base_pos) > 1.3) {
    //     reward += -0.1 * (base_pos * base_pos);
    // }


    // Alive bonus (High enough to ensure "Survival" > "Suicide")
    reward += 0.01;

    // std::cout << "The reward is " << reward << " for this angle: " << angle << " velocity: " << vel_angle << " position: " << base_pos << " and control: " << control << std::endl;
    
    return reward*0.05; 
}

void Sim::store_rollout_step(int step_idx, int env_id) {
    size_t env_offset = env_id;
    size_t step_offset_base = (size_t)step_idx * num_envs;

    // Scalars
    rollout_rewards[step_offset_base + env_offset] = global_reward_buffer[env_offset];
    rollout_values[step_offset_base + env_offset] = global_value_buffer[env_offset];
    rollout_log_probs[step_offset_base + env_offset] = global_log_prob_buffer[env_offset];
    rollout_dones[step_offset_base + env_offset] = global_done_buffer[env_offset];


    // Vectors
    size_t vec_step_offset = step_offset_base * obs_dim;
    size_t vec_env_offset = env_offset * obs_dim;
    for (int k = 0; k < obs_dim; ++k) {
        rollout_observations[vec_step_offset + vec_env_offset + k] = global_observation_buffer[vec_env_offset + k];
    }

    vec_step_offset = step_offset_base * action_dim;
    vec_env_offset = env_offset * action_dim;
    for (int k = 0; k < action_dim; ++k) {
        rollout_actions[vec_step_offset + vec_env_offset + k] = global_action_buffer[vec_env_offset + k];
    }
}

void Sim::compute_gae(int steps) {
    // Resize vectors
    rollout_advantages.resize(rollout_rewards.size());
    rollout_returns.resize(rollout_rewards.size());

    float gae_lambda = 0.95f; // Standard PPO param

    #pragma omp parallel for num_threads(n_cores_)
    for (int env_id = 0; env_id < num_envs; ++env_id) {
        float last_gae_lam = 0.0;
        
        // Get value of state T+1 (Bootstrapping)
        float next_value = global_value_buffer[env_id];

        // Loop backwards
        for (int t = steps - 1; t >= 0; --t) {
            size_t idx = (size_t)t * num_envs + env_id;
            
            float reward = rollout_rewards[idx];
            bool done = rollout_dones[idx];
            float current_value = rollout_values[idx];

            // If done, next_value is 0. If not, it's the Value of next state
            float next_non_terminal = done ? 0.0f : 1.0f;
            
            // 1. Calculate Delta (Temporal Difference Error)
            // delta = r + gamma * V(t+1) - V(t)
            float delta = reward + gamma * next_value * next_non_terminal - current_value;

            // 2. Calculate GAE
            // A_t = delta + (gamma * lambda) * A_t+1
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam;

            // Store Advantage
            rollout_advantages[idx] = last_gae_lam;

            // Store Return (Observed Return = Advantage + Value)
            rollout_returns[idx] = last_gae_lam + current_value;

            // Update next_value for the next iteration (which is t-1)
            next_value = current_value;
        }
    }
}

// Visualization Helpers
void Sim::step_inference() {
    if (!controller_) return;

    // 1. Compute Actions
    #pragma omp parallel num_threads(n_cores_)
    {
        int thread_id = omp_get_thread_num();
        if (thread_id < n_cores_) {
             controller_->computeActions(thread_id);
        }
    }

    // 2. Step Physics (We use index 0 for rollout buffers as we don't care about history here)
    // We need to resize buffers if they haven't been sized (run() usually does this)
    if (rollout_observations.empty()) {
        int steps = 1;
        rollout_observations.resize((size_t)steps * (size_t)num_envs * (size_t)obs_dim);
        rollout_actions.resize((size_t)steps * (size_t)num_envs * (size_t)action_dim);
        rollout_log_probs.resize((size_t)steps * (size_t)num_envs);
        rollout_values.resize((size_t)steps * (size_t)num_envs);
        rollout_rewards.resize((size_t)steps * (size_t)num_envs);
        rollout_returns.resize((size_t)steps * (size_t)num_envs);
        rollout_advantages.resize((size_t)steps * (size_t)num_envs);
        rollout_dones.resize((size_t)steps * (size_t)num_envs);
    }

    step_parallel(0);
}

void Sim::load_state_to_mjdata(int env_id, mjData* d) {
    // Copy from global buffer to provided mjData
    if (env_id < 0 || env_id >= num_envs) return;
    load_state_from_buffer(env_id, d, global_simstate_buffer);
}

void Sim::set_env_state(int env_id, const std::vector<double>& qpos, const std::vector<double>& qvel) {
    if (env_id >= num_envs) return;
    
    // 1. Load current state to d_[0] to preserve other fields (time, mocap, etc)
    // Note: d_[0] is used by worker threads too, but step_inference is usually called sequentially 
    // in visualization loop. If sim_loop runs in thread, we must be careful.
    // But 'keyboard' callback runs in main thread.
    // Ideally we should use a local data structure or lock.
    // Since Sim uses d_ per core, and visualization is single "environment" focused usually (index 0).
    // Let's create a temp data if we can, but we need the model. 
    // d_[0] is risky if visualization thread is running step_inference parallel to this.
    
    // For now assuming the mutex in visualize.cpp protects us?
    // In visualize.cpp:
    // keyboard() -> locks mtx -> calls g_sim->set_env_state 
    // sim_loop() -> locks mtx -> calls g_sim->step_inference
    // So they are mutually exclusive. Safe to use d_[0] as long as step_inference doesn't use d_[0] in a way that persists across calls?
    // step_parallel uses d_[thread_id]. thread 0 uses d_[0].
    // Since we are locked, step_parallel is NOT running when set_env_state is running.
    
    load_state_from_buffer(env_id, d_[0].get(), global_simstate_buffer);
    
    // 2. Modify
    for(size_t i=0; i<qpos.size() && i < (size_t)m_->nq; ++i) d_[0]->qpos[i] = qpos[i];
    for(size_t i=0; i<qvel.size() && i < (size_t)m_->nv; ++i) d_[0]->qvel[i] = qvel[i];
    
    // 3. Save back
    save_state_to_buffer(env_id, d_[0].get(), global_simstate_buffer);
}

}
