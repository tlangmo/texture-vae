# Use NVIDIA PyTorch NGC container for aarch64/ARM64 with CUDA support
# This image includes PyTorch 2+, CUDA 13.0.1, and is optimized for DGX Spark
FROM nvcr.io/nvidia/pytorch:25.09-py3

# Install system dependencies including Node.js for Claude Code CLI
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    sudo \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# # Create a non-root user with bash as default shell
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

ENV PATH="/usr/local/bin:$PATH"

# # Install Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

# # Set working directory
WORKDIR /workspace

