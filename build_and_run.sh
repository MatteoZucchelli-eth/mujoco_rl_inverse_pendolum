mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run the executable
./bin/mujoco_rl