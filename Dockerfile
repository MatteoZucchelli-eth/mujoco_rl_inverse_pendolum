# Use Ubuntu 22.04 base image
FROM ubuntu:22.04

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory in container
WORKDIR /app

# Install Python, pip, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    libgl1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libglfw3-dev \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    ninja-build \
    wget \
    unzip \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir wandb

# Install c++ PyTorch
RUN cd /tmp && \
    wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip -O libtorch.zip && \
    unzip libtorch.zip -d /app/lib && \
    rm libtorch.zip

# Build mujoco library
RUN cd /tmp && \
    git clone https://github.com/google-deepmind/mujoco.git && \
    mkdir -p /tmp/mujoco/build && \
    cd /tmp/mujoco/build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/app/lib/mujoco \
        -DCMAKE_BUILD_TYPE=Release \
        -DMUJOCO_BUILD_EXAMPLES=OFF \
        -DMUJOCO_BUILD_TESTS=OFF \
        -DMUJOCO_BUILD_SIMULATE=OFF && \
    cmake --build . -j4 && \
    cmake --install . && \
    cd / && \
    rm -rf /tmp/mujoco

# Setup bashrc to source workspace automatically
RUN echo 'export CMAKE_PREFIX_PATH=/app/lib/mujoco:$CMAKE_PREFIX_PATH' >> ~/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/app/lib/libtorch/lib:$LD_LIBRARY_PATH' >> ~/.bashrc

# Copy source files (optional for devcontainer as it mounts volume, but good for standalone build)
COPY CMakeLists.txt /app/CMakeLists.txt
COPY src/ /app/src/

