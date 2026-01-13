cmake_minimum_required(VERSION 3.8)
project(mujoco_rl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(OpenMP REQUIRED)

# Set MuJoCo paths
set(MUJOCO_DIR /app/lib/mujoco)
include_directories(include ${MUJOCO_DIR}/include)
link_directories(${MUJOCO_DIR}/lib)

add_executable(mujoco_rl
  src/main.cpp
  src/sim.cpp
)

target_link_libraries(mujoco_rl
  mujoco
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(mujoco_rl)

install (TARGETS mujoco_rl
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
