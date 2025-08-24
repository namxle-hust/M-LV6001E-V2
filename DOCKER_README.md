# Docker Setup for Multi-Modal Heterogeneous GNN

This guide provides comprehensive instructions for running the Multi-Modal Heterogeneous GNN project in Docker containers.

## 📋 Prerequisites

### Required
- Docker Engine 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- At least 16GB RAM
- 20GB+ free disk space

### Optional (for GPU support)
- NVIDIA GPU with CUDA 11.8+ support
- NVIDIA Docker Runtime ([Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repository>
cd multimodal-gnn
```

### 2. Make the Docker script executable
```bash
chmod +x docker_run.sh
```

### 3. Build the Docker image

**For GPU:**
```bash
./docker_run.sh build gpu
# OR using docker-compose:
docker-compose build multimodal-gnn-gpu
```

**For CPU:**
```bash
./docker_run.sh build cpu
# OR using docker-compose:
docker-compose build multimodal-gnn-cpu
```

### 4. Prepare your data
Place your data files in the appropriate directories:
```
data/
├── features/
│   ├── genes_expr.tsv
│   ├── genes_cnv.tsv
│   ├── cpgs.tsv
│   ├── mirnas.tsv
│   └── samples.txt
└── edges/
    ├── gene_cpg.csv
    └── gene_mirna.csv
```

### 5. Run training
```bash
# Using the shell script (GPU)
./docker_run.sh train gpu

# Using the shell script (CPU)
./docker_run.sh train cpu

# Using docker-compose
docker-compose run multimodal-gnn-gpu train

# Direct Docker command
docker run --rm --gpus all \
    -v $(pwd)/data:/workspace/multimodal_gnn/data \
    -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
    multimodal-gnn:gpu train
```

## 📚 Detailed Usage

### Using the Shell Script (`docker_run.sh`)

The provided shell script simplifies Docker operations:

```bash
# Build image
./docker_run.sh build [gpu|cpu]

# Run training with custom arguments
./docker_run.sh train gpu --epochs 200 --batch_size 64

# Run evaluation
./docker_run.sh eval gpu --checkpoint outputs/checkpoints/level1_best.pt

# Start Jupyter Lab
./docker_run.sh jupyter gpu 8888

# Start interactive shell
./docker_run.sh shell gpu

# Start TensorBoard
./docker_run.sh tensorboard 6006

# Clean up Docker resources
./docker_run.sh cleanup
```

### Using Docker Compose

Docker Compose provides more control over services:

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d multimodal-gnn-gpu

# Run training
docker-compose run multimodal-gnn-gpu train

# Run evaluation
docker-compose run multimodal-gnn-gpu eval --checkpoint outputs/checkpoints/level1_best.pt

# Start Jupyter service
docker-compose up -d jupyter

# View logs
docker-compose logs -f multimodal-gnn-gpu

# Stop all services
docker-compose down
```

### Direct Docker Commands

For maximum control:

```bash
# Build image
docker build -t multimodal-gnn:gpu --build-arg DEVICE_TYPE=gpu .

# Run training (GPU)
docker run --rm \
    --gpus all \
    -v $(pwd)/data:/workspace/multimodal_gnn/data \
    -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
    -v $(pwd)/config:/workspace/multimodal_gnn/config \
    --shm-size=16g \
    -e CUDA_VISIBLE_DEVICES=0 \
    multimodal-gnn:gpu train --epochs 100

# Run training (CPU)
docker run --rm \
    -v $(pwd)/data:/workspace/multimodal_gnn/data \
    -v $(pwd)/outputs:/workspace/multimodal_gnn/outputs \
    -v $(pwd)/config:/workspace/multimodal_gnn/config \
    --shm-size=8g \
    multimodal-gnn:cpu train --device cpu

# Interactive development
docker run --rm -it \
    --gpus all \
    -v $(pwd):/workspace/multimodal_gnn \
    -p 8888:8888 \
    -p 6006:6006 \
    multimodal-gnn:gpu bash
```

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Optional: Weights & Biases logging
WANDB_API_KEY=your_wandb_api_key

# CUDA settings
CUDA_VISIBLE_DEVICES=0

# Memory settings
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Volume Mounts

The Docker setup uses several volume mounts:

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data` | `/workspace/multimodal_gnn/data` | Input data |
| `./outputs` | `/workspace/multimodal_gnn/outputs` | Model outputs |
| `./config` | `/workspace/multimodal_gnn/config` | Configuration files |
| `./src` | `/workspace/multimodal_gnn/src` | Source code (dev only) |

### GPU Configuration

To use specific GPUs:

```bash
# Use GPU 0 only
docker run --gpus '"device=0"' ...

# Use GPUs 0 and 1
docker run --gpus '"device=0,1"' ...

# Use all GPUs
docker run --gpus all ...
```

## 📊 Monitoring

### TensorBoard

Monitor training progress with TensorBoard:

```bash
# Start TensorBoard
./docker_run.sh tensorboard 6006

# Or using docker-compose
docker-compose up -d tensorboard

# Access at http://localhost:6006
```

### Jupyter Lab

For interactive development and visualization:

```bash
# Start Jupyter Lab
./docker_run.sh jupyter gpu 8888

# Or using docker-compose
docker-compose up -d jupyter

# Access at http://localhost:8888
```

### Container Logs

View container logs:

```bash
# Using docker-compose
docker-compose logs -f multimodal-gnn-gpu

# Using docker
docker logs -f multimodal-gnn-container
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```bash
   # Reduce batch size
   ./docker_run.sh train gpu --batch_size 16
   
   # Or increase shared memory
   docker run --shm-size=32g ...
   ```

2. **Permission denied errors**
   ```bash
   # Run with user permissions
   docker run --user $(id -u):$(id -g) ...
   ```

3. **GPU not detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   
   # Install NVIDIA Container Toolkit
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Slow data loading**
   ```bash
   # Increase number of workers
   ./docker_run.sh train gpu --num_workers 8
   
   # Increase shared memory
   docker run --shm-size=32g ...
   ```

### Performance Optimization

1. **Enable mixed precision training**
   ```bash
   ./docker_run.sh train gpu --mixed_precision true
   ```

2. **Use multiple GPUs**
   ```bash
   docker run --gpus all \
       -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
       multimodal-gnn:gpu train --distributed true
   ```

3. **Optimize Docker build cache**
   ```bash
   # Use BuildKit for faster builds
   DOCKER_BUILDKIT=1 docker build -t multimodal-gnn:gpu .
   ```

## 🧹 Cleanup

Remove Docker resources when done:

```bash
# Stop and remove all containers
./docker_run.sh cleanup

# Remove images
docker rmi multimodal-gnn:gpu multimodal-gnn:cpu

# Remove all unused Docker resources
docker system prune -a

# Remove specific volumes
docker volume rm multimodal_gnn_jupyter-data
```

## 📁 Project Structure in Container

```
/workspace/multimodal_gnn/
├── config/              # Configuration files
├── data/               # Data directory (mounted)
│   ├── features/       # Feature matrices
│   └── edges/          # Edge lists
├── outputs/            # Output directory (mounted)
│   ├── checkpoints/    # Model checkpoints
│   ├── logs/           # Training logs
│   ├── tensors/        # Exported embeddings
│   └── evaluation/     # Evaluation results
├── src/                # Source code
│   ├── dataio/         # Data loading modules
│   ├── models/         # Model definitions
│   ├── losses/         # Loss functions
│   └── utils/          # Utilities
└── scripts/            # Training/evaluation scripts
```

## 🔐 Security Considerations

1. **Don't run as root in production**
   ```bash
   # Add user in Dockerfile
   RUN useradd -m -u 1000 mluser
   USER mluser
   ```

2. **Limit resources**
   ```bash
   docker run --cpus="4" --memory="16g" ...
   ```

3. **Use secrets for sensitive data**
   ```bash
   # Use Docker secrets for API keys
   echo "your_api_key" | docker secret create wandb_key -
   ```

## 📝 Additional Notes

- The Docker image includes all necessary dependencies for both CPU and GPU execution
- Model checkpoints and outputs are persisted in mounted volumes
- The container automatically creates required directories
- TensorBoard and Jupyter services can run simultaneously
- For production deployment, consider using Kubernetes or Docker Swarm

## 🆘 Support

For issues specific to Docker setup:
1. Check container logs: `docker logs <container_name>`
2. Verify volume mounts: `docker inspect <container_name>`
3. Test GPU access: `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

For model-related issues, refer to the main README.md.