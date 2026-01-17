#include <mujoco_rl/sim.hpp>

namespace mj_pool {

Sim::Sim() {
    std::cout << "Hello world"  << std::endl;
}

void Sim::init() {
    // Run sequentially once to fill the buffer
    // Ensure we have created per-core data structures. create_data()
    // will also allocate the per-core mjData objects if needed.
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
        
        // Optional: Apply initial randomization here
        
        // Save this pristine initial state (never overwritten)
        save_state_to_buffer(i, d_[0].get(), global_initial_state_buffer);
        
        // Also save to current state buffer so first step can begin
        save_state_to_buffer(i, d_[0].get(), global_simstate_buffer);
    }
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
    #pragma omp parallel
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

            // step physics
            mj_step(model, data);

            // Compute reward
            global_reward_buffer[i] = (float)compute_reward(data);

            // Check if episode is done
            bool done = (data->time > max_sim_time_); 
            global_done_buffer[i] = done; 

            // Save transition to rollout buffer (obs_t, act_t, rew_t, val_t, log_prob_t)
            store_rollout_step(step_idx, i);

            // Handle Reset or Continue
            if (done) {
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
            int qpos_limit = std::min(model->nq, obs_dim);
            
            // Copy qpos
            for (int k = 0; k < qpos_limit; k++) {
                global_observation_buffer[obs_offset + k] = (float)data->qpos[k];
            }
            
            // Copy qvel. We continue filling the buffer from where qpos left off.
            // Ensure we don't overflow obs_dim even if model has more states than we need.
            int qvel_limit = std::min(model->nv, obs_dim - qpos_limit);
            
            for (int k = 0; k < qvel_limit; k++) {
                global_observation_buffer[obs_offset + qpos_limit + k] = (float)data->qvel[k];
            }
        }
        }
    }
}

float* Sim::get_log_prob_buffer(int thread_id) {
    size_t offset = (size_t)thread_id * envs_per_thread_; // 1 float per env
    return &global_log_prob_buffer[offset];
}
float* Sim::get_value_buffer(int thread_id) {
    size_t offset = (size_t)thread_id * envs_per_thread_; // 1 float per env
    return &global_value_buffer[offset];
}
float* Sim::get_reward_buffer(int thread_id) {
    size_t offset = (size_t)thread_id * envs_per_thread_; // 1 float per env
    return &global_reward_buffer[offset];
}

float* Sim::get_observation_buffer() { return &global_observation_buffer[0]; }
float* Sim::get_action_buffer() { return &global_action_buffer[0]; }
float* Sim::get_log_prob_buffer() { return &global_log_prob_buffer[0]; }
float* Sim::get_value_buffer() { return &global_value_buffer[0]; }

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
    rollout_returns.resize((size_t)steps * (size_t)num_envs);
    rollout_advantages.resize((size_t)steps * (size_t)num_envs);
    rollout_dones.resize((size_t)steps * (size_t)num_envs);

    for (int s = 0; s < steps; ++s) {
        // Parallel inference
        #pragma omp parallel
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
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        controller_->computeActions(thread_id);
    }
   
    // Compute rewards to go (returns) after collecting all steps
    compute_returns(steps);
    compute_advantages(steps);

    // Calculate and log mean reward
    double total_reward = 0.0;
    #pragma omp parallel for reduction(+:total_reward)
    for (size_t i = 0; i < rollout_rewards.size(); ++i) {
        total_reward += (double)rollout_rewards[i];
    }
    double mean_reward = total_reward / (double)rollout_rewards.size();
    std::cout << "Mean Batch Reward: " << mean_reward << std::endl;
    
    // Train Policy and Value Function
    train();
}

void Sim::train() {
    controller_->updatePolicy(rollout_observations, rollout_actions, rollout_log_probs, rollout_returns, rollout_advantages);
}

// Reward based on angle (upright = 0) and angular velocity
double Sim::compute_reward(const mjData* d) {
    // Assuming inverted pendulum where qpos[1] is angle (or qpos[0] if single joint)
    // The goal is to be upright.
    // Let's assume standard gym inverted pendulum:
    // angle is qpos[1] (if qpos[0] is slides) or just qpos[0] depending on model.
    // For simplicity, let's assume index 0 for now as generic, but should be checked with model.
    // Typically reward = 1.0 (for staying alive) or -(theta^2 + 0.1*theta_dot^2)
    
    // Simple placeholder reward: -(angle^2 + 0.1 * velocity^2)
    // We assume qpos[0] is the angle the pole makes with vertical?
    // Or maybe cartpole? If it's pure inverted pendulum:
    double angle = d->qpos[1]; // Typically index 1 is hinge if index 0 is slider.
    // Let's check model dim. For InvertedPendulum-v4 (gym), qpos has 2 elements (slider, hinge).
    // Accessing qpos[1] safely requires checking model structure, but for now assuming it exists.
    double vel = d->qvel[1];

    // Check bounds
    if (m_->nq < 2) {
        // Fallback for simple pendulum
        angle = d->qpos[0];
        vel = d->qvel[0];
    }
    
    // Reward for keeping pole upright (angle close to 0)
    // small angle approximation or cos(angle)
    // Here using simple quadratic cost
    return -(angle * angle + 0.1 * vel * vel) + 1.0; 
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

void Sim::compute_returns(int steps) {
    rollout_returns.resize(rollout_rewards.size());

    // Compute GAE or just returns separately for each environment
    #pragma omp parallel for
    for (int env_id = 0; env_id < num_envs; ++env_id) {
        float gae = 0.0;
        
        // Initialize next_value with the value of the state AFTER the last step (s_T)
        // This handles the case where the batch ends but the episode is not done (bootstrapping).
        float next_value = global_value_buffer[env_id]; 
        
        // Iterate backwards from last step to 0
        for (int t = steps - 1; t >= 0; --t) {
            size_t idx = (size_t)t * num_envs + env_id;
            
            // Allow for non-terminal returns calculation
            // If rollout_dones[idx] is true, next value is 0 (terminal state has 0 value)
            // Otherwise it's the discounted return from t+1 + reward
            
            float reward = rollout_rewards[idx];
            bool done = rollout_dones[idx];
            
            // Standard Rewards-to-Go (Monte Carlo Returns)
            if (done) {
                next_value = 0.0;
            }
            
            next_value = reward + gamma * next_value;
            rollout_returns[idx] = next_value;
        }
    }
}

void Sim::compute_advantages(int steps) {
    // Advantage = Returns - Value
    #pragma omp parallel for
    for (size_t i = 0; i < rollout_returns.size(); ++i) {
        rollout_advantages[i] = rollout_returns[i] - rollout_values[i];
    }
}
}
