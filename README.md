# Multi-Omics Graph Neural Network

A PyTorch implementation of a heterogeneous graph neural network for multi-omics data integration. This project implements Level-1 training with unsupervised learning for patient representation learning.

## Features

- **Multi-modal Integration**: Integrates mRNA, CNV, DNA methylation, and miRNA data
- **Heterogeneous Graph Structure**: Models different relationships between omics types
- **Attention-based Fusion**: Learns importance weights for each modality
- **Unsupervised Learning**: Uses reconstruction and consistency losses
- **Two-stage Training**: Pretraining followed by full training with attention

## Requirements

```bash
pip install torch>=2.0.0
pip install torch-geometric
pip install pandas numpy scikit-learn
pip install pyyaml tqdm tensorboard
```

## Project Structure

```
project/
├── config/                 # Configuration files
│   ├── default.yaml       # Default configuration
│   ├── model.yaml         # Model-specific config
│   └── train.yaml         # Training config
├── data/                  # Data directory
│   ├── features/          # Node feature files
│   │   ├── genes.tsv      # Gene expression (mRNA + CNV)
│   │   ├── cpgs.tsv       # DNA methylation
│   │   ├── mirnas.tsv     # miRNA expression
│   │   └── samples.txt    # Patient IDs (one per line)
│   └── edges/             # Edge relationship files
│       ├── gene_cpg.csv   # CpG-to-gene mappings
│       ├── gene_mirna.csv # miRNA-to-gene targets
│       └── gene_gene.csv  # PPI (optional)
├── src/                   # Source code
│   ├── dataio/           # Data loading and processing
│   ├── models/           # Model components
│   ├── losses/           # Loss functions
│   └── train/            # Training utilities
├── scripts/              # Executable scripts
│   ├── train_level1.py   # Main training script
│   └── eval_level1.py    # Evaluation script
└── outputs/              # Output directory
    ├── checkpoints/      # Model checkpoints
    ├── logs/             # TensorBoard logs
    └── tensors/          # Saved embeddings
```

## Data Format

### Feature Files (TSV format)

All feature files should be tab-separated with:

- **Rows**: Features (genes, CpG sites, miRNAs)
- **Columns**: Patient IDs
- **Values**: Normalized and imputed expression values

Example `genes.tsv`:

```
        Patient1    Patient2    Patient3
GENE1   0.523       -0.234      1.234
GENE2   -0.123      0.456       0.789
```

### Edge Files (CSV format)

Edge files define relationships between nodes:

- **Column 1**: Source node ID
- **Column 2**: Target node ID
- **Column 3** (optional): Edge weight

Example `gene_cpg.csv`:

```
cpg_id,gene_id,weight
cg12345,GENE1,1.0
cg67890,GENE2,0.8
```

### Sample File

`samples.txt` contains one patient ID per line:

```
Patient1
Patient2
Patient3
```

## Configuration

Key configuration parameters in `config/default.yaml`:

```yaml
# Model architecture
model:
  hidden_dim: 256 # Hidden dimension size
  num_layers: 3 # Number of GNN layers
  num_heads: 4 # Attention heads
  pooling_type: 'mean' # mean or attention

# Training
training:
  batch_size: 8
  num_epochs: 200
  learning_rate: 0.001
  pretrain_epochs: 50 # Stage A epochs

# Loss weights
losses:
  feature_recon: 1.0 # Feature reconstruction
  edge_recon: 1.0 # Edge reconstruction
  consistency: 0.5 # Consistency loss
  entropy_reg: 0.1 # Entropy regularization
```

## Training

### Basic Training

```bash
python scripts/train_level1.py --config config/train.yaml
```

### Advanced Options

```bash
python scripts/train_level1.py \
    --config config/train.yaml \
    --device cuda \
    --batch_size 16 \
    --lr 0.001 \
    --epochs 300 \
    --seed 42
```

### Training Process

The training follows a two-stage protocol:

1. **Stage A (Pretraining)**:

   - Trains encoder and decoders with reconstruction losses only
   - Runs for `pretrain_epochs` epochs
   - Establishes good feature representations

2. **Stage B (Full Training)**:
   - Enables modality attention and all losses
   - Continues from pretrained weights
   - Runs until convergence with early stopping

## Evaluation

Evaluate the trained model:

```bash
python scripts/eval_level1.py \
    --checkpoint outputs/checkpoints/level1_best.pt \
    --data_split val \
    --batch_size 8
```

This will report:

- Feature reconstruction MSE per node type
- Edge prediction AUROC/AUPRC per relation
- Attention weight statistics

## Outputs

After training, the following files are saved:

### Checkpoints

- `outputs/checkpoints/level1_best.pt`: Best model checkpoint

### Embeddings

- `outputs/tensors/patient_embeddings.pt`: Final patient representations (h_i)
- `outputs/tensors/modality_embeddings.pt`: Per-modality vectors (z_i^m)

### Attention Weights

- `outputs/tensors/attention_weights.csv`: Attention weights per patient and modality

## Loading Saved Embeddings

```python
import torch

# Load patient embeddings
data = torch.load('outputs/tensors/patient_embeddings.pt')
embeddings = data['embeddings']  # Shape: (n_patients, hidden_dim)
patient_ids = data['patient_ids']

# Load attention weights
import pandas as pd
attention_df = pd.read_csv('outputs/tensors/attention_weights.csv')
```

## Using for Level-2

The saved patient embeddings (h_i) can be used for downstream tasks:

```python
# Load pretrained embeddings
checkpoint = torch.load('outputs/checkpoints/level1_best.pt')
config = checkpoint['config']

# Initialize model
model = MultiOmicsGNN(config, num_features_dict)
model.load_state_dict(checkpoint['model_state_dict'])

# Get embeddings for new data
with torch.no_grad():
    results = model(patient_graph)
    patient_embedding = results['patient_embedding']
```

## Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs/logs/
```

This shows:

- Training/validation losses
- Learning rate schedule
- Gradient norms
- Individual loss components

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Reduce `hidden_dim` or `num_layers`
- Use gradient accumulation

### Poor Convergence

- Adjust learning rate
- Modify loss weights
- Increase `pretrain_epochs`
- Check data normalization

### Missing Data

- Ensure all patients in `samples.txt` have features
- Missing values are automatically imputed with 0
- Check edge files for valid node IDs

## Citation

If you use this code, please cite:

```
[Your citation here]
```

## License

MIT License
