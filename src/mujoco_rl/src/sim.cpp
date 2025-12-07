#include <mujoco_rl/sim.hpp>

Sim::Sim() {
    std::cout << "Hello world"  << std::endl;
}

void Sim::create_model(char* filename) {
    
    // For debugging
    constexpr int error_sz = 1024;
    char error[error_sz];
    mjModel* m_raw = mj_loadXML(filename, nullptr, error, error_sz);
    if (!m_raw) {
        throw std::runtime_error(std::string(error));
    }

    m_ = MjModelPtr(m_raw);
}

void Sim::create_data() {
    int n_cores = omp_get_num_procs();
    omp_set_dynamic(0);
    omp_set_num_threads(n_cores);

    d_.resize(n_cores); // One data structure for core

    for (int i= 0; i < n_cores; i++) {
        d_[i] = MjDataPtr(mj_makeData(m_.get())); // Returned the raw pointer
    }
}