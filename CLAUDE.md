# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-modal heterogeneous graph neural network (GNN) for cancer genomics data integration. Integrates mRNA expression, CNV (Copy Number Variation), DNA methylation, and miRNA data modalities using PyTorch and PyTorch Geometric.

## Key Development Commands

### Training
```bash
# Basic training with 5-fold cross-validation (default)
python scripts/train_level1.py --config config/default.yaml

# Custom K-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 10

# Override training parameters
python scripts/train_level1.py --config config/default.yaml --batch_size 64 --lr 0.001 --epochs 300 --device cuda
```

**Important**: K-fold cross-validation is the primary training mode. The script:
1. Runs K-fold CV to evaluate model performance
2. Trains a final model on all patients
3. Exports embeddings and attention weights to `outputs/tensors/`

### Docker Commands
```bash
# Make script executable (one-time)
chmod +x docker_run.sh

# Build Docker image
./docker_run.sh build gpu  # or 'cpu'

# Run training with K-fold evaluation
./docker_run.sh train gpu --kfold 10

# Run with custom parameters
./docker_run.sh train gpu --epochs 200 --batch_size 64
```

**Note**: The Docker script has been simplified - `shell`, `jupyter`, and `cleanup` commands mentioned in old documentation may not be available. Check `docker_run.sh` for current commands.

### Dependencies Installation
```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

**Critical**: NumPy must be <2.0 for PyTorch compatibility (already pinned in requirements.txt)

## Architecture Overview

### Two-Stage Training Protocol

1. **Stage A (Pretraining)**: Trains with reconstruction losses only
   - Feature reconstruction for all modalities (mRNA, CNV, CpG methylation, miRNA)
   - Edge reconstruction with negative sampling
   - Learns robust node embeddings before fusion

2. **Stage B (Fusion)**: Adds attention-based multi-modal fusion
   - Enables cross-modality attention mechanism
   - Adds consistency loss between fused and modality-specific embeddings
   - Adds entropy regularization to prevent attention collapse

Each stage uses separate optimizers and learning rates (configurable in `config/default.yaml`).

### Core Data Flow

1. **Feature Loading** (`src/dataio/load_features.py`):
   - Loads TSV files: `genes_expr.tsv`, `genes_cnv.tsv`, `cpgs.tsv`, `mirnas.tsv`
   - Features are rows, patients are columns
   - Returns dict with node IDs and feature matrices

2. **Edge Loading** (`src/dataio/load_edges.py`):
   - Loads CSV files: `gene_cpg.csv`, `gene_mirna.csv`, optional `gene_gene.csv` (PPI)
   - Creates bidirectional edges (e.g., `cpg->gene` and `gene->cpg`)
   - Supports edge weights (defaults to 1.0)

3. **Graph Construction** (`src/dataio/build_patient_graph.py`):
   - Builds one `HeteroData` graph per patient
   - Gene nodes have 2 channels: [mRNA, CNV] concatenated
   - CpG and miRNA nodes have 1 channel each
   - Stores `gene_mrna` and `gene_cnv` separately for loss computation

4. **Batching** (`src/dataio/dataset.py`):
   - `PatientGraphDataset`: PyTorch Dataset wrapper
   - `custom_collate_fn`: Batches graphs using PyG's `Batch.from_data_list()`
   - Stores `gene_mrna_batched` and `gene_cnv_batched` as [batch_size, n_genes] tensors
   - `MultiModalDataModule`: Handles train/val/test splits and K-fold CV

### Model Architecture (`src/models/`)

1. **Encoder** (`hetero_encoder.py`):
   - Options: `HeteroGAT` (Graph Attention) or `HeteroRGCN` (Relational GCN)
   - Processes heterogeneous graph with multiple node/edge types
   - Returns per-node embeddings for gene, cpg, mirna

2. **Modality Pooling** (`modality_pool.py`):
   - Pools node embeddings to modality-level embeddings
   - For genes: separate projection heads split the 2-channel features into mRNA and CNV embeddings
   - For CpG/miRNA: aggregates node embeddings (mean or attention pooling)
   - Returns 4 modality embeddings: [mRNA, CNV, CpG methylation, miRNA]

3. **Attention Fusion** (`modality_attention.py`):
   - Only active in Stage B
   - Computes attention weights over the 4 modality embeddings
   - Returns fused patient embedding and attention weights

4. **Main Model** (`multimodal_gnn.py`):
   - `MultiModalHeteroGNN`: Encoder + Pooling + Attention
   - `MultiModalGNNWithDecoders`: Adds decoders and loss computation
   - `set_training_stage()`: Switches between Stage A and B

### Loss Functions (`src/losses/`)

1. **Feature Reconstruction** (`recon_feature.py`):
   - `MultiModalFeatureDecoders`: Separate MLP decoders for each modality
   - `FeatureReconstructionLoss`: Computes MSE per modality with configurable weights
   - Separate losses for mRNA and CNV despite sharing gene nodes

2. **Edge Reconstruction** (`recon_edge.py`):
   - Inner product decoder for link prediction
   - Negative sampling (default ratio: 5 negative edges per positive edge)
   - Computes BCE loss and AUROC/AUPRC metrics

3. **Consistency Loss** (`consistency.py`):
   - L2 distance between fused embedding and each modality embedding
   - Only active in Stage B
   - Encourages modality embeddings to be similar to the fused representation

4. **Entropy Regularization** (`consistency.py`):
   - Encourages diverse attention weights across modalities
   - Prevents attention collapse to a single modality

### Graph Structure

- **Node Types**: `gene`, `cpg`, `mirna`
- **Edge Types** (bidirectional):
  - `(cpg, maps_to, gene)` and `(gene, mapped_by, cpg)`
  - `(mirna, targets, gene)` and `(gene, targeted_by, mirna)`
  - `(gene, ppi, gene)` - optional protein-protein interactions

### Configuration System (`config/default.yaml`)

Hierarchical YAML with sections: `data`, `model`, `losses`, `training`, `logging`, `eval`

**Command-line overrides**:
- `--lr` overrides `training.stage_a.learning_rate`
- `--epochs` overrides `training.stage_a.epochs`
- `--batch_size` overrides `training.batch_size`
- `--device` overrides `device` (options: `cuda`, `cpu`, `auto`)
- `--kfold` sets number of folds for cross-validation

**Key config parameters**:
- `model.encoder.type`: `HeteroGAT` or `HeteroRGCN`
- `model.pooling.type`: `mean` or `attention`
- `losses.lambda_*`: Loss weighting coefficients
- `losses.neg_sampling_ratio`: Negative edges per positive edge
- `training.stage_a/stage_b`: Separate configs for each training stage

## Data Format Requirements

### Feature Files (TSV)
- **Format**: Features as rows, patients as columns
- **Files**: `genes_expr.tsv`, `genes_cnv.tsv`, `cpgs.tsv`, `mirnas.tsv`
- **Values**: Normalized and imputed expression/methylation values
- **Location**: `data/features/`

### Edge Files (CSV)
- **Format**: `source_id,target_id,weight` (weight column optional)
- **Files**: `gene_cpg.csv`, `gene_mirna.csv`, `gene_gene.csv` (optional)
- **Location**: `data/edges/`

### Sample File
- **File**: `data/features/samples.txt`
- **Format**: One patient ID per line (must match column names in feature files)

### Output Structure
```
outputs/
├── checkpoints/
│   ├── fold_0/, fold_1/, ...    # Per-fold checkpoints
│   └── level1_best.pt           # Final model trained on all data
├── logs/
│   ├── run_<timestamp>/         # TensorBoard logs
│   └── kfold_results.json       # K-fold CV results
└── tensors/
    ├── patient_embeddings.pt    # Final patient embeddings [N, hidden_size]
    ├── modality_embeddings.pt   # Dict with keys: mrna, cnv, cpg, mirna
    └── attention_weights.csv    # Attention weights [N, 4] for each patient
```

## Important Implementation Details

### mRNA and CNV Handling
- **During encoding**: Concatenated as 2-channel input to gene nodes
- **During pooling**: Split using separate projection heads before pooling
- **During loss**: Reconstructed separately with independent loss terms
- **Rationale**: Allows sharing structural information while maintaining modality-specific representations

### K-Fold Cross-Validation Flow
1. Data module splits patients into K folds
2. For each fold:
   - Create fresh model instance
   - Train Stage A (reconstruction only)
   - Train Stage B (with attention fusion)
   - Evaluate on validation fold
   - Save fold checkpoint
3. Aggregate metrics across folds (mean ± std)
4. Train final model on all patients (no validation set)
5. Export embeddings for all patients

### Checkpoint Contents
```python
{
    'epoch': int,
    'stage': 'A' or 'B',
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'metrics': dict,
    'config': dict
}
```

## Common Development Patterns

### Loading Saved Embeddings
```python
import torch
import pandas as pd

# Load patient embeddings
h_i = torch.load('outputs/tensors/patient_embeddings.pt')  # [N_patients, hidden_size]

# Load modality embeddings
modality_emb = torch.load('outputs/tensors/modality_embeddings.pt')
z_mrna = modality_emb['mrna']     # [N_patients, hidden_size]
z_cnv = modality_emb['cnv']
z_cpg = modality_emb['cpg']
z_mirna = modality_emb['mirna']

# Load attention weights
attn_df = pd.read_csv('outputs/tensors/attention_weights.csv', index_col=0)
# Columns: ['mrna', 'cnv', 'cpg', 'mirna']
```

### Resuming Training from Checkpoint
```python
checkpoint = torch.load('outputs/checkpoints/level1_best.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

## Key File Locations

- Training entry point: [scripts/train_level1.py](scripts/train_level1.py)
- Model definition: [src/models/multimodal_gnn.py](src/models/multimodal_gnn.py)
- Data pipeline: [src/dataio/](src/dataio/)
- Loss functions: [src/losses/](src/losses/)
- Main configuration: [config/default.yaml](config/default.yaml)
- Docker management: [docker_run.sh](docker_run.sh)

## Notes for Code Modifications

- **Adding new modalities**: Update `MultiModalFeatureDecoders`, `ModalityPooling`, `ModalityAttention` to handle new modality embeddings
- **Changing encoder**: Modify `model.encoder.type` in config; both HeteroGAT and HeteroRGCN are supported
- **Adjusting loss weights**: Edit `losses.lambda_*` coefficients in config
- **Debugging attention**: Set `eval.save_attention_weights: true` and check `outputs/tensors/attention_weights.csv`
- **Testing**: No formal test suite currently; validation is performed via K-fold CV
