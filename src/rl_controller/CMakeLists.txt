cmake_minimum_required(VERSION 3.8)
project(rl_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. Find Dependencies
find_package(ament_cmake REQUIRED)
list(APPEND CMAKE_PREFIX_PATH "/app/lib/libtorch")
find_package(Torch REQUIRED) 

# 2. Define the Library
add_library(rl_controller src/controller.cpp)

# 3. Link Dependencies
target_include_directories(rl_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(rl_controller
  ${TORCH_LIBRARIES}
)

# 4. Export dependencies
ament_export_include_directories(include)
ament_export_libraries(rl_controller)
ament_export_dependencies(Torch)

# 5. Install Rules
install(
  TARGETS rl_controller
  EXPORT export_rl_controller
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

ament_export_targets(export_rl_controller HAS_LIBRARY_TARGET)
ament_package()