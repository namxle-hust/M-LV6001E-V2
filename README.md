# Multi-Modal Heterogeneous GNN for Cancer Genomics

## Overview

This project implements a **Level 1 heterogeneous graph neural network** for multi-omics cancer data integration, supporting:
- **mRNA expression**: Gene expression levels
- **Copy Number Variation (CNV)**: Gene copy number alterations
- **DNA methylation**: CpG site methylation levels
- **miRNA expression**: microRNA expression levels

Each patient is represented as a heterogeneous graph with three node types (genes, CpG sites, miRNAs) connected by biological relationships. The model learns patient-level representations through graph neural networks and attention-based modality fusion.

## Project Structure

```
project/
├── config/
│   └── default.yaml              # Unified configuration file
├── data/
│   ├── features/                 # Patient feature tables
│   │   ├── genes_expr.tsv        # mRNA expression (TSV: features×patients)
│   │   ├── genes_cnv.tsv         # CNV data (TSV: features×patients)
│   │   ├── cpgs.tsv              # DNA methylation (TSV: CpGs×patients)
│   │   ├── mirnas.tsv            # miRNA expression (TSV: miRNAs×patients)
│   │   └── samples.txt           # Patient IDs (one per line)
│   └── edges/                    # Edge relationships (CSV)
│       ├── gene_cpg.csv          # CpG→gene mappings (cpg_id,gene_id,weight)
│       ├── gene_mirna.csv        # miRNA→gene targets (mirna_id,gene_id,weight)
│       └── gene_gene.csv         # Gene PPI (optional) (gene_id1,gene_id2,weight)
├── src/
│   ├── dataio/                   # Data loading and graph construction
│   │   ├── load_features.py      # TSV feature loading
│   │   ├── load_edges.py         # CSV edge loading with weights
│   │   ├── build_patient_graph.py # HeteroData graph construction
│   │   └── dataset.py            # PyG Dataset + K-fold DataModule
│   ├── models/                   # Neural network architectures
│   │   ├── hetero_encoder.py     # HeteroGAT/RGCN encoder
│   │   ├── modality_pool.py      # Node→modality pooling with projection heads
│   │   ├── modality_attention.py # Cross-modality attention fusion
│   │   └── multimodal_gnn.py     # Complete model with decoders
│   ├── losses/                   # Loss functions
│   │   ├── recon_feature.py      # Feature reconstruction (MSE per modality)
│   │   ├── recon_edge.py         # Edge reconstruction (BCE + negative sampling)
│   │   └── consistency.py        # Consistency loss + entropy regularization
│   └── utils/                    # Utilities
│       ├── seed.py               # Reproducibility
│       ├── logging.py            # Logging setup
│       └── kfold.py              # K-fold splitting utilities
├── scripts/
│   └── train_level1.py           # Main training script (K-fold + final model)
├── docs/
│   └── LEVEL1_ARCHITECTURE.md    # Detailed architecture documentation
├── outputs/
│   ├── checkpoints/              # Model checkpoints
│   │   ├── fold_0/, fold_1/, ... # Per-fold checkpoints
│   │   └── level1_best.pt        # Final model (trained on all data)
│   ├── logs/                     # TensorBoard logs + JSON metrics
│   │   ├── run_<timestamp>/      # TensorBoard event files
│   │   ├── kfold_results.json    # K-fold CV aggregated results
│   │   └── final_metrics.json    # Final model metrics
│   └── tensors/                  # Exported embeddings
│       ├── patient_embeddings.pt # [N_patients, hidden_size]
│       ├── modality_embeddings.pt # Dict: {mrna, cnv, cpg, mirna}
│       └── attention_weights.csv # [N_patients, 4] attention per modality
├── docker_run.sh                 # Docker management script
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Claude Code instructions
└── README.md                     # This file
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

The project uses a unified configuration file `config/default.yaml`. Key sections:

### Data Configuration
```yaml
data:
  features_dir: 'data/features/'        # Location of TSV feature files
  edges_dir: 'data/edges/'              # Location of CSV edge files
  samples_file: 'data/features/samples.txt'  # Patient IDs

  # Feature files
  gene_expr_file: 'genes_expr.tsv'
  gene_cnv_file: 'genes_cnv.tsv'
  cpg_file: 'cpgs.tsv'
  mirna_file: 'mirnas.tsv'

  # Edge files
  gene_cpg_file: 'gene_cpg.csv'
  gene_mirna_file: 'gene_mirna.csv'
  gene_gene_file: 'gene_gene.csv'       # Optional PPI

  use_ppi: false                        # Enable gene-gene PPI edges
  default_edge_weight: 1.0              # Default if weight not in CSV
```

### Model Architecture
```yaml
model:
  encoder:
    type: 'HeteroGAT'                   # Options: HeteroGAT, HeteroRGCN
    hidden_size: 256                    # Shared embedding dimension
    num_layers: 3                       # Number of GNN layers
    dropout: 0.2
    use_layer_norm: true                # Layer normalization
    use_residual: true                  # Residual connections
    heads: 4                            # Attention heads (HeteroGAT only)

  pooling:
    type: 'mean'                        # Options: mean, attention

  attention:
    hidden_size: 128                    # Attention MLP hidden size
    num_heads: 1                        # Number of attention heads
    temperature: 1.0                    # Softmax temperature
    dropout: 0.1

  decoders:                             # Feature reconstruction decoders
    gene_decoder:
      hidden_sizes: [128, 64]           # MLP hidden layers
      dropout: 0.1
    cpg_decoder:
      hidden_sizes: [128]
      dropout: 0.1
    mirna_decoder:
      hidden_sizes: [128]
      dropout: 0.1
```

### Loss Configuration
```yaml
losses:
  # Feature reconstruction weights
  lambda_recon_mrna: 1.0
  lambda_recon_cnv: 1.0
  lambda_recon_cpg: 1.0
  lambda_recon_mirna: 1.0

  # Edge reconstruction
  lambda_edge: 0.5                      # Edge reconstruction loss weight
  neg_sampling_ratio: 5                 # Negative edges per positive edge

  # Modality weighting (null = auto-compute from node counts)
  modality_weights:
    mrna: null
    cnv: null
    cpg: null
    mirna: null

  # Stage B losses
  lambda_cons: 0.1                      # Consistency loss weight
  lambda_ent: 0.01                      # Entropy regularization weight
```

### Training Configuration
```yaml
training:
  batch_size: 32
  num_workers: 4
  pin_memory: true

  # Stage A: Reconstruction pretraining
  stage_a:
    epochs: 100
    learning_rate: 0.001
    weight_decay: 1.0e-5
    scheduler: 'plateau'                # Options: plateau, cosine, step
    patience: 15                        # LR scheduler patience

  # Stage B: Attention fusion
  stage_b:
    epochs: 100
    learning_rate: 0.0001               # 10x smaller than Stage A
    weight_decay: 1.0e-5
    scheduler: 'plateau'
    patience: 15

  grad_clip: 1.0                        # Gradient clipping max norm
  early_stopping_patience: 20           # Early stopping patience

# K-fold cross-validation
kfold:
  enabled: false                        # Set by --kfold argument
  n_folds: 5                            # Number of folds
  stratified: false                     # Stratified splitting
```

### Hardware and Logging
```yaml
device: 'auto'                          # Options: cuda, cpu, auto
mixed_precision: false                  # Use FP16 training
seed: 42                                # Random seed
deterministic: true                     # Deterministic mode

logging:
  log_dir: 'outputs/logs/'
  checkpoint_dir: 'outputs/checkpoints/'
  tensors_dir: 'outputs/tensors/'
  log_interval: 10                      # Log every N batches
  save_best_only: true                  # Only save best checkpoints
  verbose: true
```

## Usage

### Training Level 1 Model

The training script performs K-fold cross-validation followed by final model training on all patients.

#### Basic Training (5-fold CV by default)

```bash
python scripts/train_level1.py --config config/default.yaml --kfold 5
```

**This will:**
1. Split patients into 5 folds
2. For each fold: Train Stage A (100 epochs) → Train Stage B (100 epochs) → Evaluate
   - Same train/val split used for both stages
   - Model continues from Stage A to Stage B (no reset)
3. Aggregate metrics across folds (mean ± std)
4. Train final model on ALL patients (Stage A → Stage B)
5. Export embeddings to `outputs/tensors/`

#### Command-Line Arguments

```bash
# Specify number of folds (default: 5)
python scripts/train_level1.py --config config/default.yaml --kfold 10

# Override config parameters
python scripts/train_level1.py \
    --config config/default.yaml \
    --kfold 5 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 200 \
    --device cuda \
    --seed 42
```

**Available overrides:**
- `--kfold N`: Number of cross-validation folds (required)
- `--batch_size N`: Override `training.batch_size`
- `--lr LR`: Override `training.stage_a.learning_rate`
- `--epochs N`: Override `training.stage_a.epochs`
- `--device DEVICE`: Override `device` (cuda/cpu/auto)
- `--seed N`: Override random seed

### Output Files

After training completes:

```
outputs/
├── checkpoints/
│   ├── fold_0/
│   │   └── checkpoint_stageB_epoch*.pt    # Best checkpoint for fold 0
│   ├── fold_1/
│   │   └── checkpoint_stageB_epoch*.pt    # Best checkpoint for fold 1
│   ├── ...
│   ├── checkpoint_stageA_epoch*.pt        # Final model Stage A checkpoints
│   ├── checkpoint_stageB_epoch*.pt        # Final model Stage B checkpoints
│   └── level1_best.pt                     # Best final model (Stage B)
├── logs/
│   ├── run_<timestamp>/                   # TensorBoard logs
│   ├── kfold_results.json                 # K-fold CV results
│   └── final_metrics.json                 # Final model metrics
└── tensors/
    ├── patient_embeddings.pt              # Shape: [N_patients, 256]
    ├── modality_embeddings.pt             # Dict with keys: mrna, cnv, cpg, mirna
    └── attention_weights.csv              # Patient attention weights
```

### Loading Saved Embeddings

```python
import torch
import pandas as pd

# Load patient embeddings (fused representation)
h_fused = torch.load('outputs/tensors/patient_embeddings.pt')
# Shape: [N_patients, 256]

# Load modality-specific embeddings
modality_emb = torch.load('outputs/tensors/modality_embeddings.pt')
z_mrna = modality_emb['mrna']      # Shape: [N_patients, 256]
z_cnv = modality_emb['cnv']        # Shape: [N_patients, 256]
z_cpg = modality_emb['cpg']        # Shape: [N_patients, 256]
z_mirna = modality_emb['mirna']    # Shape: [N_patients, 256]

# Load attention weights (Stage B)
attn_df = pd.read_csv('outputs/tensors/attention_weights.csv', index_col=0)
# Columns: ['mrna', 'cnv', 'cpg', 'mirna']
# Each row sums to 1.0 (softmax normalized)

# Example: Get attention for a specific patient
patient_id = 'Patient_001'
patient_attn = attn_df.loc[patient_id]
print(f"mRNA: {patient_attn['mrna']:.3f}, CNV: {patient_attn['cnv']:.3f}, "
      f"CpG: {patient_attn['cpg']:.3f}, miRNA: {patient_attn['mirna']:.3f}")
```

### Resuming from Checkpoint

```python
import torch
from src.models.multimodal_gnn import MultiModalGNNWithDecoders

# Load checkpoint
checkpoint = torch.load('outputs/checkpoints/level1_best.pt')

# Restore model
config = checkpoint['config']
model = MultiModalGNNWithDecoders(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Check metrics
print(f"Checkpoint from epoch {checkpoint['epoch']}, stage {checkpoint['stage']}")
print(f"Validation metrics: {checkpoint['metrics']}")
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

## Viewing Training Progress

### TensorBoard

```bash
# Launch TensorBoard to view training curves
tensorboard --logdir outputs/logs/
```

Navigate to `http://localhost:6006` to view:
- Training/validation loss curves
- Per-modality reconstruction losses
- Edge reconstruction metrics
- Learning rate schedule

### K-Fold Results

After K-fold training completes, view aggregated results:

```bash
cat outputs/logs/kfold_results.json
```

Example output:
```json
{
  "n_folds": 5,
  "averaged_metrics": {
    "val_loss": {"mean": 0.234, "std": 0.012},
    "edge_auroc": {"mean": 0.876, "std": 0.023},
    "recon_mse": {
      "mrna": {"mean": 0.045, "std": 0.003},
      "cnv": {"mean": 0.032, "std": 0.002},
      "cpg": {"mean": 0.051, "std": 0.004},
      "mirna": {"mean": 0.038, "std": 0.003}
    }
  }
}
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

## Model Architecture Summary

### Two-Stage Training Protocol

**Stage A: Reconstruction Pretraining** (100 epochs, LR=0.001)
- Learn node embeddings via graph neural networks
- Reconstruct original features (mRNA, CNV, CpG, miRNA)
- Reconstruct graph edges via link prediction
- Simple mean fusion across modalities
- **Goal**: Learn robust, biologically meaningful embeddings

**Stage B: Attention-Based Fusion** (100 epochs, LR=0.0001)
- Enable learned attention over modality embeddings
- Add consistency loss (fused ≈ modality embeddings)
- Add entropy regularization (prevent attention collapse)
- **Goal**: Learn patient-specific modality importance

### Key Architecture Components

1. **Heterogeneous Graph Encoder**
   - Options: HeteroGAT (multi-head attention) or HeteroRGCN (relational GCN)
   - 3 layers, 256 hidden dimensions, dropout 0.2
   - Residual connections + layer normalization

2. **Modality Pooling**
   - Separate projection heads for mRNA and CNV (from gene nodes)
   - Mean or attention-based pooling per node type
   - Output: 4 modality embeddings per patient [256-dim each]

3. **Cross-Modality Attention** (Stage B only)
   - MLP-based attention scoring with temperature scaling
   - Softmax-normalized weights per patient
   - Fused embedding = weighted sum of modality embeddings

4. **Feature Decoders**
   - Separate MLP decoders per modality
   - Reconstruct original node features from embeddings

5. **Edge Decoder**
   - Inner product link prediction
   - Negative sampling ratio: 5:1
   - Binary cross-entropy loss

### Loss Functions

**Stage A:**
```
L = λ_mrna·MSE(mrna) + λ_cnv·MSE(cnv) + λ_cpg·MSE(cpg) + λ_mirna·MSE(mirna)
  + λ_edge·BCE(edges)
```

**Stage B:**
```
L = L_stage_a + λ_cons·L2(fused, modalities) + λ_ent·Entropy(attention)
```

### Evaluation Metrics

- **Reconstruction**: MSE per modality (mRNA, CNV, CpG, miRNA)
- **Link Prediction**: AUROC, AUPRC for edge reconstruction
- **Validation Loss**: Total loss on held-out fold
- **Attention**: Mean weights per modality (interpretability)

## Important Notes

### Data Requirements
- All input data should be **normalized and imputed** before use
- Feature TSV files: features as rows, patients as columns
- Edge CSV files: source_id, target_id, optional weight column
- Edge weights default to 1.0 if not provided

### Model Constraints
- Requires all 4 modalities for each patient
- Graph structure is fixed (no edge learning)
- Single-patient graphs (no inter-patient edges)
- Hidden dimensions shared across all node types

### Computational Considerations
- Default batch size: 32 patients
- GPU recommended (8GB+ VRAM)
- Training time: ~30-60 min per fold on GPU (depends on dataset size)
- Mixed precision training available (`mixed_precision: true`)

### Future Extensions
- Level 2: Population-level hierarchical modeling
- Multi-task learning (survival, classification)
- Transfer learning across cancer types
- Learned edge structure

## Citation

If you use this code, please cite:

```
@software{multimodal_hetero_gnn,
  title={Multi-Modal Heterogeneous GNN for Cancer Genomics},
  year={2024}
}
```

## Docker Support

The project includes Docker support for reproducible environments with GPU/CPU options.

### Building Docker Image

```bash
# Make script executable (one-time)
chmod +x docker_run.sh

# Build GPU image (requires NVIDIA Docker)
./docker_run.sh build gpu

# Build CPU-only image
./docker_run.sh build cpu
```

### Running Training in Docker

```bash
# Basic training (default 5-fold CV)
./docker_run.sh train gpu

# Custom K-fold cross-validation
./docker_run.sh train gpu --kfold 10

# Override training parameters
./docker_run.sh train gpu --kfold 5 --batch_size 64 --epochs 200

# CPU training
./docker_run.sh train cpu --kfold 5
```

### Environment Variables

```bash
# Set Weights & Biases API key (optional)
export WANDB_API_KEY="your_key_here"
./docker_run.sh train gpu

# Specify GPU devices
export CUDA_VISIBLE_DEVICES="0,1"
./docker_run.sh train gpu
```

### Volume Mounts

The Docker container automatically mounts:
- `./data/` → `/workspace/multimodal_gnn/data/` (input data)
- `./outputs/` → `/workspace/multimodal_gnn/outputs/` (results)
- `./config/` → `/workspace/multimodal_gnn/config/` (configuration)

Results persist on the host machine after container exits.

## Additional Resources

- **Architecture Documentation**: See [docs/LEVEL1_ARCHITECTURE.md](docs/LEVEL1_ARCHITECTURE.md) for detailed technical documentation
- **Development Guide**: See [CLAUDE.md](CLAUDE.md) for Claude Code development instructions
- **Configuration Reference**: All config options documented in [config/default.yaml](config/default.yaml)
