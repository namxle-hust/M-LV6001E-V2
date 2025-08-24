# Multi-stage build for efficient image size
# Stage 1: Base image with CUDA support (for GPU)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu

# Stage 2: CPU-only base
FROM ubuntu:22.04 AS base-cpu

# Select base image based on build argument
ARG DEVICE_TYPE=gpu
FROM base-${DEVICE_TYPE} AS final-base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/workspace/.cache/torch
ENV CUDA_VISIBLE_DEVICES=0

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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Upgrade pip and install build tools
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install PyTorch (with CUDA 11.8 support for GPU, CPU version for CPU)
ARG DEVICE_TYPE=gpu
RUN if [ "$DEVICE_TYPE" = "gpu" ]; then \
    pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118; \
    else \
    pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install PyTorch Geometric and dependencies
RUN pip3 install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
    pip3 install torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
    pip3 install torch-geometric

# Install other Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt

# Copy project files
COPY . /workspace/multimodal_gnn/

# Set working directory to project root
WORKDIR /workspace/multimodal_gnn

# Create necessary directories
RUN mkdir -p data/features data/edges \
    outputs/checkpoints outputs/logs outputs/tensors \
    outputs/evaluation

# Create symbolic link for python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Set up entrypoint script
RUN echo '#!/bin/bash\n\
    if [ "$1" = "train" ]; then\n\
    echo "Starting training..."\n\
    python scripts/train_level1.py --config config/default.yaml "${@:2}"\n\
    elif [ "$1" = "eval" ]; then\n\
    echo "Starting evaluation..."\n\
    python scripts/eval_level1.py "${@:2}"\n\
    elif [ "$1" = "jupyter" ]; then\n\
    echo "Starting Jupyter Lab..."\n\
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root\n\
    elif [ "$1" = "bash" ]; then\n\
    /bin/bash\n\
    else\n\
    echo "Usage: docker run <image> [train|eval|jupyter|bash] [options]"\n\
    echo "  train: Run training script"\n\
    echo "  eval: Run evaluation script"\n\
    echo "  jupyter: Start Jupyter Lab server"\n\
    echo "  bash: Start interactive bash shell"\n\
    fi' > /workspace/entrypoint.sh && chmod +x /workspace/entrypoint.sh

# Expose ports for Jupyter and Tensorboard
EXPOSE 8888 6006

# Set entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]
CMD ["bash"]