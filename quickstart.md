# Quick Start Guide for Multi-Omics GNN

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with Sample Data

```bash
# Generate sample data and run tests
python scripts/generate_sample_data.py --n_patients 50
python scripts/test_implementation.py
```

### 3. Train Model

```bash
python scripts/train_level1.py --config config/train.yaml --epochs 100
```

## 📊 Prepare Your Data

### Required Files Structure

```
data/
├── features/
│   ├── samples.txt      # Patient IDs (one per line)
│   ├── genes.tsv        # Gene expression matrix
│   ├── cpgs.tsv         # DNA methylation matrix
│   └── mirnas.tsv       # miRNA expression matrix
└── edges/
    ├── gene_cpg.csv     # CpG-to-gene mappings
    └── gene_mirna.csv   # miRNA-to-gene targets
```

### Data Format Requirements

#### Feature Files (TSV)

- **Rows**: Features (genes/CpGs/miRNAs)
- **Columns**: Patient IDs
- **Values**: Normalized expression values

Example `genes.tsv`:

```
	Patient_001	Patient_002	Patient_003
GENE1	0.523	-0.234	1.234
GENE2	-0.123	0.456	0.789
```

#### Edge Files (CSV)

Example `gene_cpg.csv`:

```
cpg_id,gene_id,weight
cg12345678,GENE1,1.0
cg87654321,GENE2,0.8
```

## ⚙️ Configuration

### Quick Config Changes

Edit `config/train.yaml`:

```yaml
# Small model for testing
model:
  hidden_dim: 64     # Reduce for smaller GPUs
  num_layers: 2      # Fewer layers = faster

# Adjust batch size for your GPU
training:
  batch_size: 4      # Reduce if OOM

# Faster training for testing
training:
  num_epochs: 50
  pretrain_epochs: 20
```

## 🎯 Common Use Cases

### Use Case 1: Quick Test Run

```bash
# Small data, small model, few epochs
python scripts/train_level1.py \
    --config config/train.yaml \
    --batch_size 2 \
    --epochs 10 \
    --device cpu
```

### Use Case 2: Full Training on GPU

```bash
python scripts/train_level1.py \
    --config config/train.yaml \
    --device cuda \
    --batch_size 16 \
    --epochs 200
```

### Use Case 3: Evaluate Trained Model

```bash
python scripts/eval_level1.py \
    --checkpoint outputs/checkpoints/level1_best.pt \
    --data_split val
```

### Use Case 4: Extract Patient Embeddings

```python
import torch

# Load saved embeddings
data = torch.load('outputs/tensors/patient_embeddings.pt')
embeddings = data['embeddings']  # (n_patients, hidden_dim)
patient_ids = data['patient_ids']

# Use for downstream tasks
print(f"Shape: {embeddings.shape}")
print(f"Patients: {patient_ids[:5]}")
```

## 📈 Monitor Training

### TensorBoard

```bash
tensorboard --logdir outputs/logs/
```

Open browser at `http://localhost:6006`

### Check Training Progress

```python
import pandas as pd

# Load attention weights
df = pd.read_csv('outputs/tensors/attention_weights.csv')
print(df.head())

# See which modality is most important
print(df[['mRNA', 'CNV', 'DNAmeth', 'miRNA']].mean())
```

## 🔧 Troubleshooting

### Out of Memory (OOM)

```yaml
# Reduce in config/train.yaml:
model:
  hidden_dim: 32 # Smaller model
training:
  batch_size: 2 # Smaller batches
```

### Slow Training

```yaml
# Speed up training:
model:
  num_layers: 2 # Fewer layers
  num_heads: 2 # Fewer attention heads
```

### Poor Performance

```yaml
# Improve performance:
training:
  learning_rate: 0.0001 # Try different LR
  pretrain_epochs: 100 # More pretraining
losses:
  consistency: 0.2 # Adjust loss weights
```

## 📝 Key Commands Reference

| Task                 | Command                                             |
| -------------------- | --------------------------------------------------- |
| Generate sample data | `python scripts/generate_sample_data.py`            |
| Test implementation  | `python scripts/test_implementation.py`             |
| Train model          | `python scripts/train_level1.py`                    |
| Evaluate model       | `python scripts/eval_level1.py --checkpoint <path>` |
| Monitor training     | `tensorboard --logdir outputs/logs/`                |

## 💡 Tips

1. **Start Small**: Test with small data and model first
2. **Check Data**: Ensure your data is properly normalized
3. **Monitor Loss**: Watch for divergence in early epochs
4. **Save Often**: Use `checkpoint.save_frequency` to save regularly
5. **Use Pretrain**: Stage A (pretrain) helps stability

## 🆘 Getting Help

1. Check the full README.md for detailed documentation
2. Review config/default.yaml for all available options
3. Look at test_implementation.py for usage examples
4. Enable debug mode: `--debug` flag when training

## 🎉 Next Steps

After successful Level-1 training:

1. **Extract embeddings**: Use saved `patient_embeddings.pt`
2. **Analyze attention**: Check which modalities are important
3. **Downstream tasks**: Use embeddings for classification/clustering
4. **Level-2 training**: Implement supervised fine-tuning (not included)

Good luck with your multi-omics analysis! 🧬🔬
