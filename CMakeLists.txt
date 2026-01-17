cmake_minimum_required(VERSION 3.10)
project(InversePendulumTraining)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- Dependencies ---

# 1. LibTorch
list(APPEND CMAKE_PREFIX_PATH "/app/lib/libtorch")
find_package(Torch REQUIRED)

# 2. MuJoCo
# Assuming standard install or defined path. 
# Providing the path explicitly as seen in previous cmake files
set(MUJOCO_DIR /app/lib/mujoco)
include_directories(${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

# 3. OpenMP
find_package(OpenMP REQUIRED)

# --- Subdirectories ---
add_subdirectory(src/rl_controller)
add_subdirectory(src/mujoco_rl)
