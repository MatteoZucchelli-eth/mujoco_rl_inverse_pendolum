add_library(rl_controller SHARED
    src/controller.cpp
    src/network.cpp
)

target_include_directories(rl_controller PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rl_controller PUBLIC ${TORCH_LIBRARIES})
