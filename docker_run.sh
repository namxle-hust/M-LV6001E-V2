#!/bin/bash

# Simple Docker Management Script for Multi-Modal GNN
# Usage: ./docker_run.sh [command] [device] [options]

set -e

# Configuration
IMAGE_NAME="multimodal-gnn"
DEFAULT_DEVICE="gpu"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Setup required directories
setup_directories() {
    mkdir -p data/{features,edges} outputs/{checkpoints,logs,tensors} config
}

# Build Docker image
build_image() {
    local device_type=${1:-$DEFAULT_DEVICE}
    
    print_info "Building Docker image for $device_type..."
    
    # Get current user IDs for proper permissions
    local user_id=$(id -u 2>/dev/null || echo "1000")
    local group_id=$(id -g 2>/dev/null || echo "1000")
    
    docker build \
        --build-arg DEVICE_TYPE=$device_type \
        --build-arg USER_ID=$user_id \
        --build-arg GROUP_ID=$group_id \
        -t ${IMAGE_NAME}:${device_type} \
        -f Dockerfile .
    
    print_info "Docker image built successfully: ${IMAGE_NAME}:${device_type}"
}

# Get Docker run arguments based on device type
get_run_args() {
    local device_type=$1
    local args="--rm -v $(pwd)/data:/workspace/multimodal_gnn/data"
    args="$args -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs"
    args="$args -v $(pwd)/config:/workspace/multimodal_gnn/config"
    
    # GPU support
    if [ "$device_type" = "gpu" ]; then
        if docker info 2>/dev/null | grep -q "nvidia"; then
            args="$args --gpus all --shm-size=16g"
        else
            print_error "NVIDIA Docker runtime not found. Use 'cpu' instead."
            exit 1
        fi
    else
        args="$args --shm-size=8g"
    fi
    
    # Environment variables
    if [ -n "$WANDB_API_KEY" ]; then
        args="$args -e WANDB_API_KEY=$WANDB_API_KEY"
    fi
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        args="$args -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
    
    echo "$args"
}

# Run training (includes K-fold evaluation)
run_training() {
    local device_type=${1:-$DEFAULT_DEVICE}
    shift
    
    setup_directories
    print_info "Starting training with K-fold evaluation on $device_type..."
    
    local run_args=$(get_run_args $device_type)
    
    docker run $run_args ${IMAGE_NAME}:${device_type} \
        python scripts/train_level1.py --config config/default.yaml "$@"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [command] [device] [options]

Commands:
    build [gpu|cpu]              Build Docker image (default: gpu)
    train [gpu|cpu] [args]       Run training with K-fold evaluation

Devices:
    gpu                         Use GPU with CUDA support (requires NVIDIA Docker)
    cpu                         Use CPU only

Examples:
    $0 build gpu                Build GPU Docker image
    $0 build cpu                Build CPU Docker image
    $0 train gpu                Run training with default settings
    $0 train gpu --kfold 10     Run training with 10-fold cross-validation
    $0 train cpu --epochs 50    Run training on CPU for 50 epochs per stage

Environment Variables:
    WANDB_API_KEY               Weights & Biases API key
    CUDA_VISIBLE_DEVICES        GPU devices to use (e.g., "0,1")

Note: 
- Training automatically includes K-fold cross-validation with evaluation
- Directories (data/, outputs/, config/) are created automatically
- Default K-fold is 5, change with --kfold parameter
EOF
}

# Main script logic
check_docker

case "$1" in
    build)
        build_image $2
        ;;
    train)
        shift
        run_training "$@"
        ;;
    help|--help|-h|"")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac