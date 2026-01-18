add_executable(mujoco_rl 
    src/main.cpp 
    src/sim.cpp
)

target_include_directories(mujoco_rl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(mujoco_rl PRIVATE
    rl_controller
    mujoco
    OpenMP::OpenMP_CXX
)

find_package(glfw3 3.3 REQUIRED)

add_executable(visualize 
    src/visualize.cpp
    src/sim.cpp
)

target_include_directories(visualize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(visualize PRIVATE
    rl_controller
    mujoco
    OpenMP::OpenMP_CXX
    glfw
)
