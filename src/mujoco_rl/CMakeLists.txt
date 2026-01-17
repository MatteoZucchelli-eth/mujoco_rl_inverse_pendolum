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
