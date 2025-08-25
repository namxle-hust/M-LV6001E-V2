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

# Function to setup host directories with proper permissions
setup_directories() {
    print_info "Setting up host directories..."
    
    # Create directories if they don't exist
    mkdir -p data/{features,edges}
    mkdir -p outputs/{checkpoints,logs,tensors,evaluation}
    mkdir -p config
    
    # Set appropriate permissions
    chmod -R 755 data/ outputs/ config/ 2>/dev/null || {
        print_warning "Could not set permissions. You might need sudo access."
        print_info "If you encounter permission errors, run: sudo chmod -R 755 data/ outputs/ config/"
    }
    
    # Try to fix ownership to current user
    if command -v id &> /dev/null; then
        chown -R $(id -u):$(id -g) data/ outputs/ config/ 2>/dev/null || {
            print_warning "Could not change ownership. This might cause permission issues."
            print_info "If needed, run: sudo chown -R \$(id -u):\$(id -g) data/ outputs/ config/"
        }
    fi
}

# Function to build Docker image
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
        ${CUDA_VISIBLE_DEVICES:+--build-arg CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES} \
        -t ${IMAGE_NAME}:${device_type} \
        -f Dockerfile .
    
    print_info "Docker image built successfully: ${IMAGE_NAME}:${device_type}"
}

# Function to get common Docker run arguments
get_common_run_args() {
    local device_type=$1
    local interactive=${2:-false}
    local args=""
    
    # Basic args
    args="--rm"
    
    # Interactive mode
    if [ "$interactive" = "true" ]; then
        args="$args -it"
    fi
    
    # GPU support
    if [ "$device_type" = "gpu" ] && check_nvidia_docker; then
        args="$args --gpus all"
        args="$args --shm-size=16g"
    else
        args="$args --shm-size=8g"
    fi
    
    # Volume mounts
    args="$args -v $(pwd)/data:/workspace/multimodal_gnn/data"
    args="$args -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs"
    args="$args -v $(pwd)/config:/workspace/multimodal_gnn/config"
    
    # Environment variables
    if [ -n "$WANDB_API_KEY" ]; then
        args="$args -e WANDB_API_KEY=$WANDB_API_KEY"
    fi
    
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        args="$args -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    fi
    
    echo "$args"
}

# Function to run training
run_training() {
    local device_type=${1:-$DEFAULT_DEVICE}
    shift
    
    setup_directories
    print_info "Starting training on $device_type..."
    
    local run_args=$(get_common_run_args $device_type false)
    
    docker run $run_args ${IMAGE_NAME}:${device_type} train "$@"
}

# Function to run evaluation
run_evaluation() {
    local device_type=${1:-$DEFAULT_DEVICE}
    shift
    
    setup_directories
    print_info "Starting evaluation on $device_type..."
    
    local run_args=$(get_common_run_args $device_type false)
    
    docker run $run_args ${IMAGE_NAME}:${device_type} eval "$@"
}

# Function to start Jupyter Lab
run_jupyter() {
    local device_type=${1:-$DEFAULT_DEVICE}
    local port=${2:-8888}
    
    setup_directories
    print_info "Starting Jupyter Lab on port $port..."
    
    # Stop existing jupyter container if running
    docker stop ${CONTAINER_NAME}-jupyter 2>/dev/null || true
    
    local run_args=$(get_common_run_args $device_type false)
    run_args="$run_args -d --name ${CONTAINER_NAME}-jupyter -p ${port}:8888"
    
    docker run $run_args ${IMAGE_NAME}:${device_type} jupyter
    
    # Wait a moment for startup
    sleep 2
    print_info "Jupyter Lab is running at http://localhost:${port}"
    print_info "To stop: docker stop ${CONTAINER_NAME}-jupyter"
}

# Function to start interactive shell
run_shell() {
    local device_type=${1:-$DEFAULT_DEVICE}
    
    setup_directories
    print_info "Starting interactive shell..."
    
    local run_args=$(get_common_run_args $device_type true)
    
    docker run $run_args ${IMAGE_NAME}:${device_type} bash
}

# Function to start TensorBoard
run_tensorboard() {
    local port=${1:-6006}
    
    print_info "Starting TensorBoard on port $port..."
    
    # Check if logs directory exists
    if [ ! -d "outputs/logs" ]; then
        print_warning "No logs directory found. Creating empty directory."
        mkdir -p outputs/logs
    fi
    
    # Stop existing tensorboard container if running
    docker stop ${CONTAINER_NAME}-tensorboard 2>/dev/null || true
    
    docker run \
        --rm \
        -d \
        --name ${CONTAINER_NAME}-tensorboard \
        -v $(pwd)/outputs/logs:/logs:ro \
        -p ${port}:6006 \
        tensorflow/tensorflow:latest \
        tensorboard --logdir=/logs --host=0.0.0.0 --port=6006
    
    # Wait a moment for startup
    sleep 2
    print_info "TensorBoard is running at http://localhost:${port}"
    print_info "To stop: docker stop ${CONTAINER_NAME}-tensorboard"
}

# Function to show container logs
show_logs() {
    local service=${1:-""}
    
    if [ -n "$service" ]; then
        docker logs -f ${CONTAINER_NAME}-${service}
    else
        print_info "Available services:"
        docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    fi
}

# Function to show container status
show_status() {
    print_info "Container Status:"
    docker ps --filter "name=${CONTAINER_NAME}" --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" || {
        print_info "No containers running."
    }
    
    print_info "Available Images:"
    docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Function to clean up Docker resources
cleanup() {
    print_info "Cleaning up Docker resources..."
    
    # Stop running containers
    docker ps -q --filter "name=${CONTAINER_NAME}" | xargs -r docker stop
    
    # Remove stopped containers
    docker ps -aq --filter "name=${CONTAINER_NAME}" | xargs -r docker rm
    
    # Optional: Remove images (uncomment if desired)
    # docker rmi $(docker images ${IMAGE_NAME} -q) 2>/dev/null || true
    
    print_info "Cleanup complete"
}

# Function to run docker-compose commands
run_compose() {
    local action=$1
    
    if [ ! -f "docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in current directory."
        exit 1
    fi
    
    case "$action" in
        up)
            setup_directories
            print_info "Starting services with docker-compose..."
            docker-compose up -d
            print_info "Services started. Use '$0 status' to check status."
            ;;
        down)
            print_info "Stopping services with docker-compose..."
            docker-compose down
            print_info "Services stopped."
            ;;
        logs)
            docker-compose logs -f
            ;;
        *)
            print_error "Unknown compose action: $action"
            exit 1
            ;;
    esac
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
    logs [service]               Show logs for service (jupyter/tensorboard)
    status                       Show container and image status
    compose-up                   Start services using docker-compose
    compose-down                 Stop services using docker-compose
    compose-logs                 Show docker-compose logs
    cleanup                      Clean up Docker resources
    help                         Show this help message

Examples:
    $0 build gpu                 Build GPU Docker image
    $0 train gpu --epochs 100    Run training on GPU for 100 epochs
    $0 eval gpu --checkpoint outputs/checkpoints/best.pt
    $0 jupyter gpu 8890          Start Jupyter on port 8890
    $0 shell cpu                 Start CPU-based interactive shell
    $0 tensorboard 6007          Start TensorBoard on port 6007
    $0 logs jupyter              Show Jupyter container logs
    $0 status                    Show all container status

Environment Variables:
    WANDB_API_KEY               Weights & Biases API key
    CUDA_VISIBLE_DEVICES        GPU devices to use (e.g., "0,1")

Note: Directories (data/, outputs/, config/) will be created automatically
      with proper permissions when needed.
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
    logs)
        show_logs $2
        ;;
    status)
        show_status
        ;;
    compose-up)
        run_compose up
        ;;
    compose-down)
        run_compose down
        ;;
    compose-logs)
        run_compose logs
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        if [ -z "$1" ]; then
            show_usage
        else
            print_error "Unknown command: $1"
            show_usage
            exit 1
        fi
        ;;
esac