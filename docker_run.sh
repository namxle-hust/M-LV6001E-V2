#!/bin/bash

# Multi-Modal GNN Docker Management Script
# Usage: ./docker_run.sh [command] [options]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="multimodal-gnn"
CONTAINER_NAME="multimodal-gnn-container"
DEFAULT_DEVICE="gpu"

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
}

# Function to check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        print_warning "NVIDIA Docker runtime not found. GPU support may not be available."
        return 1
    fi
    return 0
}

# Function to build Docker image
build_image() {
    local device_type=${1:-$DEFAULT_DEVICE}
    
    print_info "Building Docker image for $device_type..."
    
    docker build \
        --build-arg DEVICE_TYPE=$device_type \
        -t ${IMAGE_NAME}:${device_type} \
        -f Dockerfile .
    
    print_info "Docker image built successfully: ${IMAGE_NAME}:${device_type}"
}

# Function to run training
run_training() {
    local device_type=${1:-$DEFAULT_DEVICE}
    shift
    
    print_info "Starting training on $device_type..."
    
    if [ "$device_type" = "gpu" ]; then
        docker run \
            --rm \
            --gpus all \
            -v $(pwd)/data:/workspace/multimodal_gnn/data \
            -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
            -v $(pwd)/config:/workspace/multimodal_gnn/config \
            --shm-size=16g \
            ${IMAGE_NAME}:${device_type} \
            train "$@"
    else
        docker run \
            --rm \
            -v $(pwd)/data:/workspace/multimodal_gnn/data \
            -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
            -v $(pwd)/config:/workspace/multimodal_gnn/config \
            --shm-size=8g \
            ${IMAGE_NAME}:${device_type} \
            train --device cpu "$@"
    fi
}

# Function to run evaluation
run_evaluation() {
    local device_type=${1:-$DEFAULT_DEVICE}
    shift
    
    print_info "Starting evaluation on $device_type..."
    
    if [ "$device_type" = "gpu" ]; then
        docker run \
            --rm \
            --gpus all \
            -v $(pwd)/data:/workspace/multimodal_gnn/data \
            -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
            -v $(pwd)/config:/workspace/multimodal_gnn/config \
            ${IMAGE_NAME}:${device_type} \
            eval "$@"
    else
        docker run \
            --rm \
            -v $(pwd)/data:/workspace/multimodal_gnn/data \
            -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
            -v $(pwd)/config:/workspace/multimodal_gnn/config \
            ${IMAGE_NAME}:${device_type} \
            eval --device cpu "$@"
    fi
}

# Function to start Jupyter Lab
run_jupyter() {
    local device_type=${1:-$DEFAULT_DEVICE}
    local port=${2:-8888}
    
    print_info "Starting Jupyter Lab on port $port..."
    
    if [ "$device_type" = "gpu" ]; then
        docker run \
            --rm \
            -d \
            --name ${CONTAINER_NAME}-jupyter \
            --gpus all \
            -v $(pwd):/workspace/multimodal_gnn \
            -p ${port}:8888 \
            ${IMAGE_NAME}:${device_type} \
            jupyter
    else
        docker run \
            --rm \
            -d \
            --name ${CONTAINER_NAME}-jupyter \
            -v $(pwd):/workspace/multimodal_gnn \
            -p ${port}:8888 \
            ${IMAGE_NAME}:${device_type} \
            jupyter
    fi
    
    print_info "Jupyter Lab is running at http://localhost:${port}"
    print_info "To stop: docker stop ${CONTAINER_NAME}-jupyter"
}

# Function to start interactive shell
run_shell() {
    local device_type=${1:-$DEFAULT_DEVICE}
    
    print_info "Starting interactive shell..."
    
    if [ "$device_type" = "gpu" ]; then
        docker run \
            --rm \
            -it \
            --gpus all \
            -v $(pwd):/workspace/multimodal_gnn \
            --shm-size=16g \
            ${IMAGE_NAME}:${device_type} \
            bash
    else
        docker run \
            --rm \
            -it \
            -v $(pwd):/workspace/multimodal_gnn \
            --shm-size=8g \
            ${IMAGE_NAME}:${device_type} \
            bash
    fi
}

# Function to start TensorBoard
run_tensorboard() {
    local port=${1:-6006}
    
    print_info "Starting TensorBoard on port $port..."
    
    docker run \
        --rm \
        -d \
        --name ${CONTAINER_NAME}-tensorboard \
        -v $(pwd)/outputs/logs:/logs:ro \
        -p ${port}:6006 \
        tensorflow/tensorflow:latest \
        tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    
    print_info "TensorBoard is running at http://localhost:${port}"
    print_info "To stop: docker stop ${CONTAINER_NAME}-tensorboard"
}

# Function to clean up Docker resources
cleanup() {
    print_info "Cleaning up Docker resources..."
    
    # Stop running containers
    docker ps -q --filter "name=${CONTAINER_NAME}" | xargs -r docker stop
    
    # Remove stopped containers
    docker ps -aq --filter "name=${CONTAINER_NAME}" | xargs -r docker rm
    
    print_info "Cleanup complete"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [command] [options]

Commands:
    build [gpu|cpu]              Build Docker image (default: gpu)
    train [gpu|cpu] [args]       Run training script
    eval [gpu|cpu] [args]        Run evaluation script
    jupyter [gpu|cpu] [port]     Start Jupyter Lab (default port: 8888)
    shell [gpu|cpu]              Start interactive bash shell
    tensorboard [port]           Start TensorBoard (default port: 6006)
    compose-up                   Start services using docker-compose
    compose-down                 Stop services using docker-compose
    cleanup                      Clean up Docker resources
    help                         Show this help message

Examples:
    $0 build gpu                 Build GPU Docker image
    $0 train gpu --epochs 100    Run training on GPU for 100 epochs
    $0 eval gpu --checkpoint outputs/checkpoints/best.pt
    $0 jupyter gpu 8890          Start Jupyter on port 8890
    $0 shell cpu                 Start CPU-based interactive shell
    $0 tensorboard 6007          Start TensorBoard on port 6007

Environment Variables:
    WANDB_API_KEY               Weights & Biases API key (optional)
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
    eval)
        shift
        run_evaluation "$@"
        ;;
    jupyter)
        run_jupyter $2 $3
        ;;
    shell)
        run_shell $2
        ;;
    tensorboard)
        run_tensorboard $2
        ;;
    compose-up)
        print_info "Starting services with docker-compose..."
        docker-compose up -d
        print_info "Services started. Check docker-compose ps for status."
        ;;
    compose-down)
        print_info "Stopping services with docker-compose..."
        docker-compose down
        print_info "Services stopped."
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        show_usage
        exit 1
        ;;
esac