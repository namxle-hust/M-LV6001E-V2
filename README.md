# Multi-Modal Heterogeneous GNN for Cancer Genomics

## Overview

This project implements a heterogeneous graph neural network for multi-omics cancer data integration, supporting mRNA expression, CNV, DNA methylation, and miRNA data modalities.

## Project Structure

```
project/
├── config/
│   ├── default.yaml        # Default configuration
│   ├── model.yaml          # Model architecture config
│   └── train.yaml          # Training configuration
├── data/
│   ├── features/           # Patient feature tables
│   │   ├── genes_expr.tsv  # mRNA expression data
│   │   ├── genes_cnv.tsv   # CNV data
│   │   ├── cpgs.tsv        # DNA methylation data
│   │   ├── mirnas.tsv      # miRNA expression data
│   │   └── samples.txt     # Patient IDs (one per line)
│   └── edges/              # Edge relationships
│       ├── gene_cpg.csv    # CpG→gene mappings
│       ├── gene_mirna.csv  # miRNA→gene targets
│       └── gene_gene.csv   # Gene-gene PPI (optional)
├── src/
│   ├── dataio/
│   │   ├── load_features.py      # Feature loading utilities
│   │   ├── load_edges.py         # Edge loading utilities
│   │   ├── build_patient_graph.py # Graph construction
│   │   ├── dataset.py            # PyG Dataset implementation
│   │   └── collate.py            # Custom collate function
│   ├── models/
│   │   ├── hetero_encoder.py     # HeteroGAT/RGCN encoder
│   │   ├── decoders.py           # Feature decoders
│   │   ├── modality_pool.py      # Per-modality pooling
│   │   └── modality_attention.py # Attention fusion
│   ├── losses/
│   │   ├── recon_feature.py      # Feature reconstruction
│   │   ├── recon_edge.py         # Edge reconstruction
│   │   ├── consistency.py        # Consistency loss
│   │   ├── entropy_reg.py        # Entropy regularization
│   │   └── weighting.py          # Per-modality weighting
│   ├── train/
│   │   ├── trainer.py            # Training loop
│   │   └── scheduler.py          # Learning rate scheduling
│   └── utils/
│       ├── seed.py               # Reproducibility utilities
│       ├── logging.py            # Logging setup
│       └── metrics.py            # Evaluation metrics
├── scripts/
│   └── train_level1.py           # Level-1 training script with K-fold evaluation
├── outputs/
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training logs
│   └── tensors/                  # Exported embeddings
└── README.md
```

## Data Format

### Feature Tables (TSV format)

All feature tables should have:

- Rows: Feature IDs (gene IDs, CpG IDs, miRNA IDs)
- Columns: Patient IDs
- Values: Normalized and imputed expression/methylation values

#### Example: `genes_expr.tsv`

```
        Patient1    Patient2    Patient3
BRCA1   0.523      -0.234      1.234
TP53    1.234       0.567      -0.890
...
```

#### Example: `genes_cnv.tsv`

```
        Patient1    Patient2    Patient3
BRCA1   0          1           -1
TP53    0          0            2
...
```

### Edge Files (CSV format)

#### Example: `gene_cpg.csv`

```
cpg_id,gene_id,weight
cg00000029,BRCA1,1.0
cg00000165,TP53,0.8
...
```

#### Example: `gene_mirna.csv`

```
mirna_id,gene_id,weight
hsa-mir-21,PTEN,0.9
hsa-mir-155,TP53,0.7
...
```

### Sample Definition

`samples.txt`: One patient ID per line

```
Patient1
Patient2
Patient3
...
```

## Configuration

The project uses a single unified configuration file `config/default.yaml` with hierarchical sections:

```yaml
# Model architecture
model:
  encoder:
    type: 'HeteroGAT'  # Options: HeteroGAT, HeteroRGCN
    hidden_size: 256
    num_layers: 3
    dropout: 0.2
    use_layer_norm: true
    use_residual: true

  pooling:
    type: 'mean'  # Options: mean, attention

  attention:
    hidden_size: 128
    num_heads: 1
    temperature: 1.0
    dropout: 0.1

# Loss configuration
losses:
  lambda_recon_mrna: 1.0
  lambda_recon_cnv: 1.0
  lambda_recon_cpg: 1.0
  lambda_recon_mirna: 1.0
  lambda_edge: 0.5
  lambda_cons: 0.1
  lambda_ent: 0.01
  neg_sampling_ratio: 5

# Training parameters
training:
  batch_size: 32
  stage_a:
    epochs: 100
    learning_rate: 0.001
    weight_decay: 1e-5
  stage_b:
    epochs: 100
    learning_rate: 0.0001
    weight_decay: 1e-5
  early_stopping_patience: 20
  grad_clip: 1.0

# Data configuration
data:
  features_dir: 'data/features/'
  edges_dir: 'data/edges/'
  use_ppi: false
  default_edge_weight: 1.0
```

## Usage

### Training Level-1

```bash
# Basic training
python scripts/train_level1.py --config config/default.yaml

# With custom parameters
python scripts/train_level1.py \
    --config config/default.yaml \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 300

# K-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 5

# Training with specific device
python scripts/train_level1.py --config config/default.yaml --device cuda
```

### Note on Evaluation

Evaluation is automatically performed during training with K-fold cross-validation. Each fold is trained and evaluated, with final metrics averaged across all folds. No separate evaluation step is needed.

### Loading Saved Embeddings

```python
import torch

# Load patient embeddings
h_i = torch.load('outputs/tensors/patient_embeddings.pt')

# Load modality embeddings
modality_emb = torch.load('outputs/tensors/modality_embeddings.pt')
z_mrna = modality_emb['mRNA']
z_cnv = modality_emb['CNV']
z_dnameth = modality_emb['DNAmeth']
z_mirna = modality_emb['miRNA']

# Load attention weights
attention_weights = pd.read_csv('outputs/tensors/attention_weights.csv')
```

## Key Features

### Multi-Modal Integration

- **mRNA and CNV**: Concatenated as channels on gene nodes, split during pooling using separate projection heads
- **DNA methylation**: Mapped to genes via CpG→gene edges
- **miRNA**: Connected to genes via miRNA→gene target edges

### Heterogeneous Graph Structure

- **Node Types**: `gene`, `cpg`, `mirna`
- **Edge Types**:
  - `('cpg', 'maps_to', 'gene')` - CpG sites mapped to genes
  - `('mirna', 'targets', 'gene')` - miRNA targeting relationships
  - `('gene', 'mapped_by', 'cpg')` - Reverse CpG mappings
  - `('gene', 'targeted_by', 'mirna')` - Reverse miRNA targets
  - `('gene', 'ppi', 'gene')` (optional) - Protein-protein interactions

**Edge Weights**: The system supports edge weights loaded from CSV files. If no weight column is provided, edges default to weight 1.0.

### Training Protocol

1. **Stage A (Pretrain)**: Train with reconstruction losses only

   - Separate mRNA and CNV reconstruction losses
   - Feature reconstruction for all node types
   - Edge reconstruction with negative sampling

2. **Stage B (Fusion)**: Add modality attention and consistency loss
   - Enable modality-level attention
   - Add consistency and entropy regularization
   - Continue training with full loss

### Loss Components

1. **Feature Reconstruction**: MSE per node type
2. **Edge Reconstruction**: Binary cross-entropy with negative sampling
3. **Consistency Loss**: Distance between fused and modality embeddings
4. **Entropy Regularization**: Prevent attention collapse
5. **Contrastive Regularization**: Optional robustness term

## K-Fold Cross-Validation

The framework supports K-fold cross-validation for robust evaluation.

### Running K-fold Training

```bash
# 5-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 5

# 10-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 10
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy < 2.0 (for PyTorch compatibility)
- Pandas >= 1.5.0
- scikit-learn >= 1.2.0
- PyYAML >= 6.0
- tqdm >= 4.65.0
- matplotlib, seaborn, plotly (visualization)
- tensorboard, wandb (logging)
- jupyter, jupyterlab (development)

## Installation

```bash
# Install PyTorch with CUDA 11.8 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric

# Install other dependencies
pip install -r requirements.txt
```

## Key Features

### Two-Stage Training
- **Stage A (Pretraining)**: Reconstruction losses only for robust feature learning
- **Stage B (Fusion)**: Adds cross-modality attention and consistency regularization

### Advanced Loss Functions
- Feature reconstruction with automatic modality weighting
- Edge reconstruction with configurable negative sampling
- Consistency loss between fused and modality-specific embeddings
- Entropy regularization to prevent attention collapse

### Robust Evaluation
- K-fold cross-validation with stratified splitting
- Attention weight analysis and visualization
- Comprehensive metrics (MSE, MAE, AUROC, AUPRC)
- Embedding export for downstream analysis

## Important Notes

- All input data should be normalized and imputed before use
- Edge weights are supported and default to 1.0 if not provided in CSV files
- The model uses shared hidden dimensions across node types
- Automatic modality weighting based on node counts if not manually specified
- Level-2 hierarchical modeling is not implemented in this version

## Citation

If you use this code, please cite:

```
@software{multimodal_hetero_gnn,
  title={Multi-Modal Heterogeneous GNN for Cancer Genomics},
  year={2024}
}
```

## Docker Support

The project includes simple Docker support with GPU/CPU options:

```bash
# Make Docker script executable
chmod +x docker_run.sh

# Build Docker image for GPU
./docker_run.sh build gpu

# Build for CPU
./docker_run.sh build cpu

# Run training with K-fold evaluation
./docker_run.sh train gpu                    # Default 5-fold
./docker_run.sh train gpu --kfold 10         # 10-fold cross-validation
./docker_run.sh train cpu --epochs 50        # CPU training
```
