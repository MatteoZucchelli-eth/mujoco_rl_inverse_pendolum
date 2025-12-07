# Use ROS2 Humble base image (or change to your ROS2 version)
FROM ros:humble-ros-base

# Set working directory in container
WORKDIR /app

# Install Python, pip, and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-colcon-common-extensions \
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
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir numpy

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

# Setup bashrc to source ROS2 and workspace automatically
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /app/install/setup.bash" >> ~/.bashrc

# Copy naro_msgs package files
COPY src/ /app/src/

