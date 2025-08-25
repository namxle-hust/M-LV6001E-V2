# Multi-stage build for efficient image size
# Define device type argument globally (must be before FROM instructions)
ARG DEVICE_TYPE=gpu

# Stage 1: Base image with CUDA support (for GPU)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu

# Stage 2: CPU-only base
FROM ubuntu:22.04 AS base-cpu

# Select base image based on build argument
FROM base-${DEVICE_TYPE} AS final-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/workspace/.cache/torch

# Make CUDA_VISIBLE_DEVICES configurable (default to all GPUs)
ARG CUDA_VISIBLE_DEVICES=""
ENV CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    libgomp1 \
    sudo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash appuser && \
    echo "appuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Create workspace and set permissions
RUN mkdir -p /workspace && chown -R appuser:appuser /workspace
WORKDIR /workspace

# Switch to non-root user
USER appuser

# Upgrade pip and install build tools
RUN python3 -m pip install --user --upgrade pip setuptools wheel

# Add user pip bin to PATH
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Re-declare ARG for this stage (required for multi-stage builds)
ARG DEVICE_TYPE=gpu
RUN if [ "$DEVICE_TYPE" = "gpu" ]; then \
    pip3 install --user torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118; \
    else \
    pip3 install --user torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install PyTorch Geometric conditionally based on device type
# Re-declare ARG for this section
ARG DEVICE_TYPE=gpu
RUN if [ "$DEVICE_TYPE" = "gpu" ]; then \
    pip3 install --user torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
    pip3 install --user torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html; \
    else \
    pip3 install --user torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cpu.html && \
    pip3 install --user torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html; \
    fi && \
    pip3 install --user torch-geometric

# Copy requirements first for better layer caching
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --user -r requirements.txt

# Copy project files
COPY --chown=appuser:appuser . /workspace/multimodal_gnn/

# Set working directory to project root
WORKDIR /workspace/multimodal_gnn

# Create necessary directories with proper permissions
RUN mkdir -p data/features data/edges \
    outputs/checkpoints outputs/logs outputs/tensors \
    outputs/evaluation && \
    chown -R appuser:appuser data outputs

# Create symbolic link for python3
USER root
RUN ln -sf /usr/bin/python3 /usr/bin/python
USER appuser

# Set up entrypoint script with better error handling and directory creation
RUN echo '#!/bin/bash\n\
    set -e  # Exit on error\n\
    \n\
    # Function to ensure output directories exist with proper permissions\n\
    setup_directories() {\n\
    echo "Setting up output directories..."\n\
    mkdir -p outputs/{checkpoints,logs,tensors,evaluation}\n\
    mkdir -p data/{features,edges}\n\
    \n\
    # Try to fix permissions if possible\n\
    chmod -R 755 outputs/ 2>/dev/null || echo "Warning: Could not modify permissions for outputs/"\n\
    chmod -R 755 data/ 2>/dev/null || echo "Warning: Could not modify permissions for data/"\n\
    }\n\
    \n\
    show_usage() {\n\
    echo "Usage: docker run <image> [train|eval|jupyter|bash] [options]"\n\
    echo "  train: Run training script"\n\
    echo "  eval: Run evaluation script"\n\
    echo "  jupyter: Start Jupyter Lab server"\n\
    echo "  bash: Start interactive bash shell"\n\
    }\n\
    \n\
    # Always setup directories before running commands\n\
    setup_directories\n\
    \n\
    case "$1" in\n\
    train)\n\
    echo "Starting training..."\n\
    exec python scripts/train_level1.py --config config/default.yaml "${@:2}"\n\
    ;;\n\
    eval)\n\
    echo "Starting evaluation..."\n\
    exec python scripts/eval_level1.py "${@:2}"\n\
    ;;\n\
    jupyter)\n\
    echo "Starting Jupyter Lab..."\n\
    # More secure jupyter configuration\n\
    exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser \\\n\
    --ServerApp.token="" --ServerApp.password="" \\\n\
    --ServerApp.allow_origin="*" --ServerApp.disable_check_xsrf=True\n\
    ;;\n\
    bash)\n\
    exec /bin/bash\n\
    ;;\n\
    "")\n\
    echo "No command specified. Starting bash shell."\n\
    exec /bin/bash\n\
    ;;\n\
    *)\n\
    echo "Unknown command: $1"\n\
    show_usage\n\
    exit 1\n\
    ;;\n\
    esac' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import torch_geometric; print('Health check passed')" || exit 1

# Expose ports for Jupyter and Tensorboard
EXPOSE 8888 6006

# Set entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]

# Add labels for better container management
ARG DEVICE_TYPE=gpu
LABEL maintainer="your-email@example.com" \
    version="1.0" \
    description="Multimodal GNN training environment" \
    device_type="${DEVICE_TYPE}"