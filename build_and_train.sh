mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run the executable
./bin/mujoco_rl /workspaces/inverse_pendolum_training/checkpoints_5/actor_170.pt &