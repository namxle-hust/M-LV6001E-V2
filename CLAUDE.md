# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-modal heterogeneous graph neural network (GNN) for cancer genomics data integration. The project supports mRNA expression, CNV (Copy Number Variation), DNA methylation, and miRNA data modalities, using PyTorch and PyTorch Geometric.

## Key Development Commands

### Training
```bash
# Basic Level-1 training
python scripts/train_level1.py --config config/default.yaml

# Training with custom parameters
python scripts/train_level1.py --config config/default.yaml --batch_size 64 --lr 0.001 --epochs 300

# K-fold cross-validation (default: 5-fold)
python scripts/train_level1.py --config config/default.yaml --kfold 5

# Training with specific device
python scripts/train_level1.py --config config/default.yaml --device cuda
```

### Evaluation
```bash
# Evaluate trained model
python scripts/eval_level1.py --checkpoint outputs/checkpoints/level1_best.pt --data_dir data/

# Custom evaluation
python scripts/eval_level1.py --checkpoint outputs/checkpoints/level1_best.pt --data_dir data/ --output_dir outputs/evaluation/ --batch_size 32
```

### Docker Commands
```bash
# Make Docker script executable
chmod +x docker_run.sh

# Build Docker image for GPU
./docker_run.sh build gpu

# Build for CPU
./docker_run.sh build cpu

# Run training in Docker
./docker_run.sh train gpu --epochs 200 --batch_size 64

# Start interactive shell
./docker_run.sh shell gpu

# Start Jupyter Lab
./docker_run.sh jupyter gpu 8888

# Cleanup Docker resources
./docker_run.sh cleanup
```

### Dependencies
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

## Code Architecture

### Core Components

1. **Data Pipeline** (`src/dataio/`):
   - `load_features.py`: Loads TSV feature files (genes_expr.tsv, genes_cnv.tsv, cpgs.tsv, mirnas.tsv)
   - `load_edges.py`: Loads CSV edge files (gene_cpg.csv, gene_mirna.csv)
   - `build_patient_graph.py`: Constructs heterogeneous graphs for each patient
   - `dataset.py`: PyTorch Geometric dataset implementation with custom collate functions

2. **Model Architecture** (`src/models/`):
   - `multimodal_gnn.py`: Main model class that orchestrates all components
   - `hetero_encoder.py`: Heterogeneous encoder (HeteroGAT or HeteroRGCN)
   - `modality_pool.py`: Per-modality pooling strategies (mean/attention-based)
   - `modality_attention.py`: Cross-modality attention mechanism for fusion

3. **Loss Functions** (`src/losses/`):
   - `recon_feature.py`: Feature reconstruction losses (MSE per node type)
   - `recon_edge.py`: Edge reconstruction with negative sampling
   - `consistency.py`: Consistency loss between fused and modality embeddings

4. **Training Framework**:
   - Two-stage training: Stage A (reconstruction pretraining) → Stage B (attention fusion)
   - K-fold cross-validation support with stratified splits
   - Automatic early stopping and learning rate scheduling

### Graph Structure

- **Node Types**: `gene`, `cpg`, `mirna`
- **Edge Types**:
  - `('cpg', 'maps_to', 'gene')`: CpG sites mapped to genes
  - `('mirna', 'targets', 'gene')`: miRNA targeting relationships
  - `('gene', 'ppi', 'gene')`: Protein-protein interactions (optional)

### Data Format Requirements

- **Feature files**: TSV format with features as rows, patients as columns
- **Edge files**: CSV format with source_id, target_id, weight columns
- **Sample file**: `samples.txt` with one patient ID per line
- All data should be normalized and imputed before use

### Configuration System

- Main config: `config/default.yaml`
- Hierarchical YAML structure with model, training, data, and logging sections
- Command-line overrides supported with dot notation (e.g., `--lr` overrides `training.stage_a.learning_rate`)

### Output Structure

```
outputs/
├── checkpoints/        # Model checkpoints (.pt files)
├── logs/              # TensorBoard logs and training logs
├── tensors/           # Exported embeddings and attention weights
└── evaluation/        # Evaluation reports and metrics
```

## Development Notes

- The project uses PyTorch Geometric's `HeteroData` for graph representation
- Multi-modal fusion is achieved through learnable attention over modality-specific embeddings  
- Training supports both GPU and CPU execution with automatic device detection
- Docker setup provides isolated environment with CUDA 11.8+ support
- Model checkpoints include full state for resuming training
- K-fold validation ensures robust performance evaluation across patient splits

## Important File Locations

- Training entry point: `scripts/train_level1.py`
- Model definition: `src/models/multimodal_gnn.py`
- Main configuration: `config/default.yaml`
- Docker management: `docker_run.sh`
- Requirements: `requirements.txt`