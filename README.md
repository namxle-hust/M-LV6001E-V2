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
│   ├── train_level1.py           # Level-1 training script
│   └── eval_level1.py            # Evaluation script
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

### Model Configuration (`config/model.yaml`)

```yaml
encoder:
  type: 'HeteroGAT' # or "HeteroRGCN"
  hidden_size: 256
  num_layers: 3
  dropout: 0.2
  use_layer_norm: true

pooling:
  type: 'mean' # or "attention"

attention:
  hidden_size: 128
  temperature: 1.0

decoders:
  hidden_sizes: [128, 64]

losses:
  lambda_recon: 1.0
  lambda_cons: 0.1
  lambda_ent: 0.01
  lambda_contractive: 0.001
```

### Training Configuration (`config/train.yaml`)

```yaml
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 200
  patience: 20

  stage_a:
    epochs: 100
    lr: 0.001

  stage_b:
    epochs: 100
    lr: 0.0001

data:
  num_workers: 4
  pin_memory: true

logging:
  log_interval: 10
  save_interval: 20
```

## Usage

### Training Level-1

```bash
# Basic training
python scripts/train_level1.py --config config/train.yaml

# With custom parameters
python scripts/train_level1.py \
    --config config/train.yaml \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 300
```

### Evaluation

```bash
python scripts/eval_level1.py \
    --checkpoint outputs/checkpoints/level1_best.pt \
    --data_dir data/
```

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
  - `('cpg', 'maps_to', 'gene')`
  - `('mirna', 'targets', 'gene')`
  - `('gene', 'ppi', 'gene')` (optional)

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

### Running K-fold training

````bash
# Run 5-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 5

# Run 10-fold cross-validation
python scripts/train_level1.py --config config/default.yaml --kfold 10

## Requirements

- Python 3.10+
- PyTorch 2.0+
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn
- PyYAML
- tqdm

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install pandas numpy scikit-learn pyyaml tqdm
````

## Notes

- All input data should be normalized and imputed before use
- Edge weights default to 1.0 if not provided
- The model uses shared hidden dimensions across node types
- Level-2 hierarchical modeling is not implemented in this version

## Citation

If you use this code, please cite:

```
@software{multimodal_hetero_gnn,
  title={Multi-Modal Heterogeneous GNN for Cancer Genomics},
  year={2024}
}
```
