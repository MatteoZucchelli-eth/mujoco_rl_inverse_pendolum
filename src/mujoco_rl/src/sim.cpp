#include <mujoco_rl/sim.hpp>
#include <cstdlib>  // for rand() and RAND_MAX

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
        save_initial_state(i, d_[0].get());
        
        // Also save to current state buffer so first step can begin
        save_simstate(i, d_[0].get());
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
}

void Sim::create_data() {
    if (!m_) {
        throw std::runtime_error("Model not created before create_data(). Call create_model() first.");
    }

    if (state_dim_ <= 0) {
        throw std::runtime_error("state_dim_ is not set or invalid. Ensure create_model() ran successfully.");
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
    global_simstate_buffer.resize((size_t)num_envs * (size_t)state_dim_);
    global_initial_state_buffer.resize((size_t)num_envs * (size_t)state_dim_);  // Separate buffer for pristine initial states
    global_done_buffer.resize((size_t)num_envs);

    std::cout << "Created data for device with number of cores: " << n_cores_ <<std::endl;
    std::cout << "Each core  will execute: " << envs_per_thread_ << " simulations" << std::endl;
    
}

// This function saves the state for a specific element of a batch
void Sim::save_simstate(int env_id, mjData* d) {
    int offset = env_id * state_dim_;
    double* dst = &global_simstate_buffer[offset];

    double* ptr = dst;

    // 1. Time (1)
    *ptr = d->time; 
    ptr++;

    // 2. QPOS (nq)
    mju_copy(ptr, d->qpos, m_->nq); 
    ptr += m_->nq;

    // 3. QVEL (nv)
    mju_copy(ptr, d->qvel, m_->nv); 
    ptr += m_->nv;

    // 4. ACT (na) - Only if model has actuators/muscles
    if (m_->na > 0) {
        mju_copy(ptr, d->act, m_->na); 
        ptr += m_->na;
    }
    
    // 5. MOCAP (if any)
    if (m_->nmocap > 0) {
        mju_copy(ptr, d->mocap_pos, 3 * m_->nmocap); 
        ptr += 3 * m_->nmocap;

        mju_copy(ptr, d->mocap_quat, 4 * m_->nmocap); 
        ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(ptr, d->userdata, m_->nuserdata); 
        ptr += m_->nuserdata;
    }

    // 7. Warm Start (Crucial for speed!)
    mju_copy(ptr, d->qacc_warmstart, m_->nv); 
    ptr += m_->nv;
}

// This function loads the state for a specific element of a batch
void Sim::load_simstate(int env_id, mjData* d) {
    int offset = env_id * state_dim_;
    double* src = &global_simstate_buffer[offset];

    double* ptr = src;

    // 1. Time (1)
    d->time = *ptr; 
    ptr++;

    // 2. QPOS (nq)
    mju_copy(d->qpos, ptr, m_->nq); 
    ptr += m_->nq;

    // 3. QVEL (nv)
    mju_copy(d->qvel, ptr, m_->nv); 
    ptr += m_->nv;

    // 4. ACT (na) - Only if model has actuators/muscles
    if (m_->na > 0) {
        mju_copy(d->act, ptr, m_->na); 
        ptr += m_->na;
    }
    
    // 5. MOCAP (if any)
    if (m_->nmocap > 0) {
        mju_copy(d->mocap_pos, ptr, 3 * m_->nmocap); 
        ptr += 3 * m_->nmocap;

        mju_copy(d->mocap_quat, ptr, 4 * m_->nmocap); 
        ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(d->userdata, ptr, m_->nuserdata); 
        ptr += m_->nuserdata;
    }

    // 7. Warm Start (Crucial for speed!)
    mju_copy(d->qacc_warmstart, ptr, m_->nv); 
    ptr += m_->nv;
}

// Save pristine initial state (called once per environment in init())
void Sim::save_initial_state(int env_id, mjData* d) {
    int offset = env_id * state_dim_;
    double* dst = &global_initial_state_buffer[offset];

    double* ptr = dst;

    // 1. Time (1)
    *ptr = d->time; 
    ptr++;

    // 2. QPOS (nq)
    mju_copy(ptr, d->qpos, m_->nq); 
    ptr += m_->nq;

    // 3. QVEL (nv)
    mju_copy(ptr, d->qvel, m_->nv); 
    ptr += m_->nv;

    // 4. ACT (na) - Only if model has actuators/muscles
    if (m_->na > 0) {
        mju_copy(ptr, d->act, m_->na); 
        ptr += m_->na;
    }
    
    // 5. MOCAP (if any)
    if (m_->nmocap > 0) {
        mju_copy(ptr, d->mocap_pos, 3 * m_->nmocap); 
        ptr += 3 * m_->nmocap;

        mju_copy(ptr, d->mocap_quat, 4 * m_->nmocap); 
        ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(ptr, d->userdata, m_->nuserdata); 
        ptr += m_->nuserdata;
    }

    // 7. Warm Start (Crucial for speed!)
    mju_copy(ptr, d->qacc_warmstart, m_->nv); 
    ptr += m_->nv;
}

// Load pristine initial state (used on reset)
void Sim::load_initial_state(int env_id, mjData* d) {
    int offset = env_id * state_dim_;
    double* src = &global_initial_state_buffer[offset];

    double* ptr = src;

    // 1. Time (1)
    d->time = *ptr; 
    ptr++;

    // 2. QPOS (nq)
    mju_copy(d->qpos, ptr, m_->nq); 
    ptr += m_->nq;

    // 3. QVEL (nv)
    mju_copy(d->qvel, ptr, m_->nv); 
    ptr += m_->nv;

    // 4. ACT (na) - Only if model has actuators/muscles
    if (m_->na > 0) {
        mju_copy(d->act, ptr, m_->na); 
        ptr += m_->na;
    }
    
    // 5. MOCAP (if any)
    if (m_->nmocap > 0) {
        mju_copy(d->mocap_pos, ptr, 3 * m_->nmocap); 
        ptr += 3 * m_->nmocap;

        mju_copy(d->mocap_quat, ptr, 4 * m_->nmocap); 
        ptr += 4 * m_->nmocap;
    }

    // 6. User Data
    if (m_->nuserdata > 0) {
        mju_copy(d->userdata, ptr, m_->nuserdata); 
        ptr += m_->nuserdata;
    }

    // 7. Warm Start (Crucial for speed!)
    mju_copy(d->qacc_warmstart, ptr, m_->nv); 
    ptr += m_->nv;
}

void Sim::add_noise(mjData* d) {
    
    // Add noise to each position
    for (int i = 0; i < m_->nq; i++) {
        // Generate random number in [0, 1] and scale to [noise_min, noise_max]
        double noise = noise_min + (static_cast<double>(rand()) / RAND_MAX) * (noise_max - noise_min);
        d->qpos[i] += noise;
    }
}
void Sim::step_parallel() {
    #pragma omp parallel

    {
        int thread_id = omp_get_thread_num();
        mjData* data = d_[thread_id].get(); // My private workbench
        mjModel* model = m_.get();

        int start_index = thread_id * envs_per_thread_;
        int end_index = start_index + envs_per_thread_;

        // Loop thrugh the batch assigned to this thread
        for (int i = start_index; i < end_index; i++)  {
            // stop if we exceed  the total number of environments. This can happen if num_envs is 
            // not perfectly divisible by n_cores_
            if (i >= num_envs) break;
            // load the state from the buffer
            load_simstate(i, data);

            // apply action
            int action_offset = i * action_dim;
            for (int k=0; k < action_dim; k++) {
                data->ctrl[k] = global_action_buffer[action_offset + k];
            }

            // step physics
            mj_step(model, data);

            // Check if episode is done
            bool done = (data->time > max_sim_time_); // for now let's just check this. Eventually we can also add here with a or a terminal  condition given by the user

            // IMPORTANT: Save observation BEFORE reset (this is the terminal observation if done)
            // This follows proper RL convention where the returned observation corresponds to the done flag
            int obs_offset = i * obs_dim;
            for (int k = 0; k < obs_dim; k++) {
                global_observation_buffer[obs_offset + k] = data->qpos[k];
            }

            // Save the done flag
            global_done_buffer[i] = done;
            
            // Save the current state (terminal state if done, regular state otherwise)
            save_simstate(i, data);

            // NOW handle reset if needed
            if (done) {
                // Load the pristine initial state
                load_initial_state(i, data);

                // Add randomization to the initial state
                add_noise(data);

                // Forward kinematics to ensure consistency
                mj_forward(model, data);
                
                // Save this new starting state to the current state buffer
                // so the next iteration starts from this randomized initial state
                save_simstate(i, data);
            }
        }
    }

}

