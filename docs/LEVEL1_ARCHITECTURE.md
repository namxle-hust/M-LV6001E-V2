# Level 1: Multi-Modal Heterogeneous Graph Neural Network Architecture

## Table of Contents
1. [Overview](#overview)
2. [Architecture Components](#architecture-components)
3. [Data Representation](#data-representation)
4. [Model Pipeline](#model-pipeline)
5. [Training Protocol](#training-protocol)
6. [Loss Functions](#loss-functions)
7. [Implementation Details](#implementation-details)
8. [Mathematical Formulation](#mathematical-formulation)

---

## Overview

The Level 1 model is a multi-modal heterogeneous graph neural network (GNN) designed for integrating cancer genomics data across four modalities:
- **mRNA expression**: Gene expression levels
- **Copy Number Variation (CNV)**: Gene copy number alterations
- **DNA methylation**: CpG site methylation levels
- **miRNA expression**: microRNA expression levels

### Key Innovation

The model treats each patient as a **heterogeneous graph** with three node types (genes, CpG sites, miRNAs) connected by biological relationships. It learns to:
1. Encode node-level information through graph convolutions
2. Pool nodes into modality-specific embeddings
3. Fuse modalities via learned attention weights
4. Reconstruct original features to ensure biological meaningfulness

### Design Philosophy

- **Structure-aware**: Leverages biological networks (CpG→gene mappings, miRNA→gene targets)
- **Modality-specific processing**: Each modality has its own pooling and decoding pathway
- **Attention-based fusion**: Learns which modalities are most informative for each patient
- **Self-supervised**: Uses reconstruction and consistency losses for training without labels

---

## Architecture Components

### 1. Input Layer

**Node Types and Features:**

| Node Type | Feature Dimension | Description |
|-----------|------------------|-------------|
| `gene` | 2 channels | Concatenated [mRNA, CNV] per gene |
| `cpg` | 1 channel | DNA methylation β-value per CpG site |
| `mirna` | 1 channel | miRNA expression value |

**Edge Types:**
- `(cpg, maps_to, gene)`: CpG sites mapped to genes via genomic coordinates
- `(gene, mapped_by, cpg)`: Reverse mapping (bidirectional)
- `(mirna, targets, gene)`: miRNA targeting relationships from databases
- `(gene, targeted_by, mirna)`: Reverse targeting (bidirectional)
- `(gene, ppi, gene)`: Optional protein-protein interactions

**Edge Weights:** Loaded from CSV files, default to 1.0 if not provided.

---

### 2. Heterogeneous Graph Encoder

**Purpose:** Transform raw node features into rich embeddings that capture both node attributes and graph structure.

#### 2.1 HeteroGAT (Graph Attention Network)

**Architecture:**
```
Input Projection (per node type) → Multi-Layer GAT → Output Embeddings

For each layer:
  - Multi-head attention (4 heads in intermediate layers, 1 head in final layer)
  - Layer normalization (optional)
  - ReLU activation
  - Dropout
  - Residual connections (optional)
```

**Key Features:**
- **Multi-head attention**: Intermediate layers use 4 attention heads, concatenated and flattened
- **Final layer**: Single attention head produces final embedding dimension
- **Heterogeneous convolution**: Different attention weights for each edge type
- **Aggregation**: Sum aggregation across incoming edges of different types

**Input/Output:**
- Input: `x_dict = {'gene': [N_genes, 2], 'cpg': [N_cpg, 1], 'mirna': [N_mirna, 1]}`
- Output: `h_dict = {'gene': [N_genes, D], 'cpg': [N_cpg, D], 'mirna': [N_mirna, D]}`
  - Where `D = hidden_size` (default: 256)

**Attention Mechanism:**
For edge type `(src, rel, dst)`:
```
α_{ij} = softmax_j(LeakyReLU(a^T [W·h_i || W·h_j]))
h_i^(l+1) = σ(Σ_j α_{ij} · W^(l) · h_j^(l))
```

#### 2.2 HeteroRGCN (Relational Graph Convolutional Network)

**Architecture:**
```
Input Projection → Multi-Layer RGCN → Output Embeddings

For each layer:
  - Relational convolution with basis decomposition (optional)
  - Layer normalization (optional)
  - ReLU activation
  - Dropout
  - Residual connections (optional)
```

**Key Features:**
- **Relation-specific parameters**: Each edge type has its own transformation
- **Basis decomposition**: Optional parameter sharing via basis decomposition
- **Mean aggregation**: Averages incoming messages across edge types

**Transformation:**
```
h_i^(l+1) = σ(Σ_r Σ_{j∈N_i^r} (1/|N_i^r|) W_r^(l) · h_j^(l) + W_0^(l) · h_i^(l))
```
Where `r` indexes edge types (relations).

**Configuration:**
- Set `model.encoder.type: 'HeteroGAT'` or `'HeteroRGCN'` in config
- Default parameters: 3 layers, 256 hidden size, 0.2 dropout

---

### 3. Modality Pooling Layer

**Purpose:** Aggregate node-level embeddings into patient-level modality representations.

**Challenge:** Gene nodes carry both mRNA and CNV information (2 channels), but we need separate modality embeddings for attention-based fusion.

**Solution:** Separate projection heads before pooling.

#### 3.1 Projection Heads

```python
# Gene embeddings: [N_genes, 256]
gene_emb = node_embeddings['gene']

# Apply separate projections
mrna_emb = linear_mrna(gene_emb)  # [N_genes, 256]
cnv_emb = linear_cnv(gene_emb)    # [N_genes, 256]
```

#### 3.2 Pooling Strategies

**Mean Pooling (default):**
```python
z_modality = mean(node_embeddings, dim=nodes)
```

**Attention Pooling:**
```python
attention_scores = MLP(node_embeddings)  # [N_nodes, 1]
attention_weights = softmax(attention_scores)
z_modality = Σ(attention_weights * node_embeddings)
```

**Batching:** When processing multiple patients, uses `torch_geometric.nn.global_mean_pool()` with batch assignments to pool per-patient.

#### 3.3 Output

Four modality embeddings per patient:
```python
modality_embeddings = {
    'mrna': [batch_size, 256],   # From gene nodes via mrna_projection
    'cnv': [batch_size, 256],    # From gene nodes via cnv_projection
    'cpg': [batch_size, 256],    # From cpg nodes
    'mirna': [batch_size, 256]   # From mirna nodes
}
```

---

### 4. Modality Attention Layer

**Purpose:** Learn patient-specific importance weights for each modality and produce a unified patient embedding.

**Only active in Stage B training.**

#### 4.1 Attention Score Computation

```python
# For each modality embedding z_m ∈ R^256
score_m = MLP(z_m)  # R^256 → R^128 → R^1

# Stack scores for all modalities
scores = [score_mrna, score_cnv, score_cpg, score_mirna]  # [batch_size, 4]

# Softmax with temperature scaling
α = softmax(scores / τ)  # [batch_size, 4]
```

**MLP Architecture:**
- Layer 1: Linear(256, 128) + LayerNorm + ReLU + Dropout(0.1)
- Layer 2: Linear(128, 1)

**Temperature:** Default τ = 1.0 (configurable)

#### 4.2 Weighted Fusion

```python
# Stack modality embeddings: [batch_size, 4, 256]
Z = stack([z_mrna, z_cnv, z_cpg, z_mirna], dim=1)

# Apply attention weights: [batch_size, 4, 1] ⊗ [batch_size, 4, 256]
h_fused = Σ_m (α_m * z_m)  # [batch_size, 256]
```

#### 4.3 Output

- **Fused embedding**: `h_fused` ∈ R^[batch_size × 256]
- **Attention weights**: `α` ∈ R^[batch_size × 4]
- **Attention dictionary**: Mean attention per modality across batch (for logging)

**Interpretation:** Higher attention weight → that modality is more important for this patient's representation.

---

### 5. Feature Decoders

**Purpose:** Reconstruct original features from node embeddings to ensure the model preserves biological information.

#### 5.1 Decoder Architecture

Each decoder is a small MLP:
```
Input Embedding [D] → Hidden Layer(s) [128, 64] → Output Feature [1]
```

**Components per layer:**
- Linear transformation
- LayerNorm
- ReLU activation
- Dropout (0.1)

#### 5.2 Modality-Specific Decoders

| Decoder | Input | Output | Purpose |
|---------|-------|--------|---------|
| `mrna_decoder` | Gene embeddings [N_genes, 256] | mRNA values [N_genes, 1] | Reconstruct mRNA expression |
| `cnv_decoder` | Gene embeddings [N_genes, 256] | CNV values [N_genes, 1] | Reconstruct copy number |
| `cpg_decoder` | CpG embeddings [N_cpg, 256] | Methylation values [N_cpg, 1] | Reconstruct β-values |
| `mirna_decoder` | miRNA embeddings [N_mirna, 256] | miRNA values [N_mirna, 1] | Reconstruct expression |

**Key Design:** Gene embeddings are decoded separately for mRNA and CNV, allowing the model to disentangle these modalities despite shared encoding.

---

### 6. Edge Decoder

**Purpose:** Reconstruct graph structure to ensure embeddings preserve relational information.

#### 6.1 Inner Product Decoder

For edge type `(src_type, relation, dst_type)`:
```python
# Get embeddings
z_src = node_embeddings[src_type]  # [N_src, 256]
z_dst = node_embeddings[dst_type]  # [N_dst, 256]

# Positive edges (actual edges in graph)
edge_index = data[edge_type].edge_index  # [2, N_edges]
src_emb = z_src[edge_index[0]]  # [N_edges, 256]
dst_emb = z_dst[edge_index[1]]  # [N_edges, 256]

# Compute scores via inner product
pos_scores = sigmoid(sum(src_emb * dst_emb, dim=-1))  # [N_edges]
```

#### 6.2 Negative Sampling

```python
# Sample random negative edges (non-existent edges)
neg_edge_index = sample_negative_edges(
    num_src_nodes,
    num_dst_nodes,
    num_samples=neg_sampling_ratio * num_positive_edges
)

# Compute negative scores
neg_scores = sigmoid(sum(z_src[neg_src] * z_dst[neg_dst], dim=-1))
```

**Default:** 5 negative edges per positive edge (`neg_sampling_ratio=5`)

#### 6.3 Loss Computation

```python
# Binary cross-entropy
pos_loss = -log(pos_scores).mean()
neg_loss = -log(1 - neg_scores).mean()
edge_loss = (pos_loss + neg_loss) / 2
```

**Metrics:** Also computes AUROC and AUPRC for edge prediction quality.

---

## Data Representation

### Patient Graph Structure

Each patient is represented as a `HeteroData` object:

```python
data = HeteroData()

# Node features
data['gene'].x = [N_genes, 2]      # [mRNA, CNV] channels
data['cpg'].x = [N_cpg, 1]         # Methylation values
data['mirna'].x = [N_mirna, 1]     # miRNA expression

# Edge connectivity
data['cpg', 'maps_to', 'gene'].edge_index = [2, N_cpg_gene_edges]
data['mirna', 'targets', 'gene'].edge_index = [2, N_mirna_gene_edges]

# Metadata
data.patient_id = "Patient_001"
data.patient_idx = 0

# Separate storage for loss computation
data.gene_mrna = [N_genes]   # Separate mRNA vector
data.gene_cnv = [N_genes]    # Separate CNV vector
```

### Batching

Multiple patients are batched using PyTorch Geometric's `Batch.from_data_list()`:

```python
batched_data = Batch.from_data_list([patient_1, patient_2, ...])

# Creates:
batched_data['gene'].batch = [0, 0, ..., 1, 1, ..., 2, 2, ...]  # Patient assignments
batched_data.gene_mrna_batched = [batch_size, N_genes]  # Stacked mRNA
batched_data.gene_cnv_batched = [batch_size, N_genes]   # Stacked CNV
```

---

## Model Pipeline

### Complete Forward Pass

```python
def forward(data):
    # 1. Extract features
    x_dict = {
        'gene': data['gene'].x,      # [N_genes_total, 2]
        'cpg': data['cpg'].x,        # [N_cpg_total, 1]
        'mirna': data['mirna'].x     # [N_mirna_total, 1]
    }

    # 2. Extract edges
    edge_index_dict = {
        ('cpg', 'maps_to', 'gene'): data['cpg', 'maps_to', 'gene'].edge_index,
        ('gene', 'mapped_by', 'cpg'): data['gene', 'mapped_by', 'cpg'].edge_index,
        ('mirna', 'targets', 'gene'): data['mirna', 'targets', 'gene'].edge_index,
        ('gene', 'targeted_by', 'mirna'): data['gene', 'targeted_by', 'mirna'].edge_index
    }

    # 3. Encode nodes
    h_dict = encoder(x_dict, edge_index_dict)
    # h_dict = {'gene': [N_genes, 256], 'cpg': [N_cpg, 256], 'mirna': [N_mirna, 256]}

    # 4. Pool to modality embeddings
    batch_dict = {
        'gene': data['gene'].batch,
        'cpg': data['cpg'].batch,
        'mirna': data['mirna'].batch
    }
    z_dict = modality_pooling(h_dict, batch_dict)
    # z_dict = {'mrna': [B, 256], 'cnv': [B, 256], 'cpg': [B, 256], 'mirna': [B, 256]}

    # 5. Fuse modalities (Stage B only)
    if stage == 'B':
        h_fused, attention_weights = modality_attention(z_dict)
        # h_fused: [B, 256], attention_weights: [B, 4]
    else:
        h_fused = mean(z_dict.values())  # Simple averaging for Stage A
        attention_weights = None

    # 6. Decode features (for loss computation)
    reconstructed = {
        'gene_mrna': mrna_decoder(h_dict['gene']),   # [N_genes, 1]
        'gene_cnv': cnv_decoder(h_dict['gene']),     # [N_genes, 1]
        'cpg': cpg_decoder(h_dict['cpg']),           # [N_cpg, 1]
        'mirna': mirna_decoder(h_dict['mirna'])      # [N_mirna, 1]
    }

    return {
        'node_embeddings': h_dict,
        'modality_embeddings': z_dict,
        'fused_embedding': h_fused,
        'attention_weights': attention_weights,
        'reconstructed': reconstructed
    }
```

---

## Training Protocol

### Two-Stage Training Strategy

#### Stage A: Reconstruction Pretraining

**Goal:** Learn robust node embeddings that preserve feature and structural information.

**Active Components:**
- Encoder (HeteroGAT/RGCN)
- Modality pooling (mean pooling across modalities)
- Feature decoders
- Edge decoder

**Loss Function:**
```
L_A = λ_mrna·L_recon(mRNA) + λ_cnv·L_recon(CNV)
    + λ_cpg·L_recon(CpG) + λ_mirna·L_recon(miRNA)
    + λ_edge·L_edge_recon
```

**Training Details:**
- Learning rate: 0.001 (default)
- Optimizer: Adam with weight decay 1e-5
- Epochs: 100 (default)
- Early stopping: patience 20
- LR scheduler: ReduceLROnPlateau (factor=0.5, patience=15)

**Output:** Pretrained encoder with biologically meaningful embeddings.

---

#### Stage B: Attention-Based Fusion

**Goal:** Learn to fuse modalities via attention and maintain consistency.

**Active Components:**
- All components from Stage A
- Modality attention (enabled)
- Consistency loss
- Entropy regularization

**Loss Function:**
```
L_B = L_A + λ_cons·L_consistency + λ_ent·L_entropy
```

Where:
- `L_consistency`: Ensures fused embedding is similar to modality embeddings
- `L_entropy`: Prevents attention collapse to a single modality

**Training Details:**
- Learning rate: 0.0001 (10× smaller than Stage A)
- Optimizer: Fresh Adam optimizer
- Epochs: 100 (default)
- Early stopping: patience 20
- LR scheduler: ReduceLROnPlateau

**Output:** Final model with learned attention fusion.

---

### K-Fold Cross-Validation

**Workflow:**

1. **Split patients into K folds** (default K=5)
   - Uses `sklearn.model_selection.KFold`
   - Stratification optional (not used by default)

2. **For each fold:**
   ```python
   for fold in range(K):
       # a. Create fresh model
       model = MultiModalGNNWithDecoders(config)

       # b. Get train/val loaders
       train_loader = data_module.train_dataloader()
       val_loader = data_module.val_dataloader()

       # c. Train Stage A
       model.set_training_stage('A')
       train_stage_a(model, train_loader, val_loader)

       # d. Train Stage B
       model.set_training_stage('B')
       train_stage_b(model, train_loader, val_loader)

       # e. Evaluate on validation fold
       metrics = evaluate_fold(model, val_loader)

       # f. Save fold checkpoint
       save_checkpoint(model, fold=fold)
   ```

3. **Aggregate results:**
   ```python
   avg_metrics = {
       'val_loss': mean ± std across folds,
       'edge_auroc': mean ± std across folds,
       'recon_mse': {
           'mrna': mean ± std,
           'cnv': mean ± std,
           'cpg': mean ± std,
           'mirna': mean ± std
       }
   }
   ```

4. **Final model training:**
   - Train on ALL patients (no validation split)
   - Export embeddings for downstream analysis

**Why K-fold?**
- Robust performance estimation with limited data
- Ensures model generalizes across different patient subsets
- Provides confidence intervals (mean ± std)

---

## Loss Functions

### 1. Feature Reconstruction Loss

**Purpose:** Ensure node embeddings preserve original feature information.

**Formula:**
```
L_recon = (1/N) Σ_i ||f_i - f̂_i||²
```

**Per-Modality MSE:**
```python
L_mrna = MSE(mrna_decoder(h_gene), original_mrna)
L_cnv = MSE(cnv_decoder(h_gene), original_cnv)
L_cpg = MSE(cpg_decoder(h_cpg), original_cpg)
L_mirna = MSE(mirna_decoder(h_mirna), original_mirna)
```

**Weighted Sum:**
```
L_recon_total = λ_mrna·L_mrna + λ_cnv·L_cnv + λ_cpg·L_cpg + λ_mirna·L_mirna
```

**Default weights:** All λ = 1.0

**Optional Modality Weighting:**
If `losses.modality_weights` are `null`, auto-compute based on inverse node counts:
```python
w_modality = N_total / N_modality
# Normalize: w_modality /= sum(weights)
```

---

### 2. Edge Reconstruction Loss

**Purpose:** Preserve graph structure in learned embeddings.

**Positive Edge Loss:**
```
L_pos = -(1/|E|) Σ_{(u,v)∈E} log(σ(z_u^T z_v))
```

**Negative Edge Loss:**
```
L_neg = -(1/|E_neg|) Σ_{(u,v)∈E_neg} log(1 - σ(z_u^T z_v))
```

**Total Edge Loss:**
```
L_edge = (L_pos + L_neg) / 2
```

**Negative Sampling:** For each positive edge, sample 5 random non-edges (configurable).

**Metrics:**
- **AUROC**: Area under ROC curve for edge classification
- **AUPRC**: Area under Precision-Recall curve

---

### 3. Consistency Loss (Stage B Only)

**Purpose:** Ensure fused embedding is consistent with individual modality embeddings.

**Formula:**
```
L_cons = (1/M) Σ_m ||h_fused - z_m||²
```

Where:
- `h_fused`: Fused patient embedding
- `z_m`: Modality-specific embedding
- `M = 4` (number of modalities)

**Normalization:** Embeddings are L2-normalized before computing distance.

**Intuition:** Fused representation should be a meaningful combination of modalities, not drastically different.

**Weight:** Default λ_cons = 0.1

---

### 4. Entropy Regularization (Stage B Only)

**Purpose:** Prevent attention weights from collapsing to a single modality.

**Formula:**
```
L_ent = -(1/B) Σ_b Σ_m α_m^(b) · log(α_m^(b))
```

Where:
- `α_m^(b)`: Attention weight for modality `m` in patient `b`
- `B`: Batch size

**Intuition:** Maximize entropy of attention distribution → encourage diverse attention.

**Weight:** Default λ_ent = 0.01

**Note:** Negative sign because we want to maximize entropy (add negative entropy as loss).

---

### 5. Total Loss

**Stage A:**
```
L_total = L_recon_total + λ_edge · L_edge
```

**Stage B:**
```
L_total = L_recon_total + λ_edge · L_edge + λ_cons · L_cons + λ_ent · L_ent
```

**Default Hyperparameters:**
```yaml
losses:
  lambda_recon_mrna: 1.0
  lambda_recon_cnv: 1.0
  lambda_recon_cpg: 1.0
  lambda_recon_mirna: 1.0
  lambda_edge: 0.5
  lambda_cons: 0.1
  lambda_ent: 0.01
  neg_sampling_ratio: 5
```

---

## Implementation Details

### Gradient Flow

**Gradient Clipping:** Max norm = 1.0
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why:** Prevents exploding gradients, especially with heterogeneous graph structures.

---

### Regularization Techniques

1. **Dropout:** Applied after each layer (default: 0.2)
2. **Layer Normalization:** Stabilizes training in deep encoders
3. **Weight Decay:** L2 regularization on parameters (1e-5)
4. **Early Stopping:** Stops if validation loss doesn't improve for 20 epochs

---

### Memory Optimization

**PyTorch Geometric Batching:**
- Efficiently batches variable-size graphs
- Sparse edge storage (COO format)
- Batch indexing for per-patient pooling

**Batch Size:** Default 32 patients (configurable)

**Pin Memory:** Enabled for faster GPU transfer

---

### Device Handling

**Automatic Device Selection:**
```python
if config['device'] == 'auto':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

**Mixed Precision:** Optional FP16 training (set `mixed_precision: true`)

---

### Checkpointing

**Checkpoint Contents:**
```python
{
    'epoch': current_epoch,
    'stage': 'A' or 'B',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'metrics': {...},
    'config': {...}
}
```

**Saved Files:**
- `outputs/checkpoints/fold_X/checkpoint_stageB_epochYYY.pt`
- `outputs/checkpoints/level1_best.pt` (final model)

---

### Embedding Export

**Exported Files:**

1. **Patient Embeddings** (`patient_embeddings.pt`):
   ```python
   h_i = torch.load('outputs/tensors/patient_embeddings.pt')
   # Shape: [N_patients, 256]
   ```

2. **Modality Embeddings** (`modality_embeddings.pt`):
   ```python
   z_dict = torch.load('outputs/tensors/modality_embeddings.pt')
   # z_dict = {'mrna': [N, 256], 'cnv': [N, 256], 'cpg': [N, 256], 'mirna': [N, 256]}
   ```

3. **Attention Weights** (`attention_weights.csv`):
   ```csv
   patient_id,mrna,cnv,cpg,mirna
   Patient_001,0.35,0.25,0.20,0.20
   Patient_002,0.40,0.30,0.15,0.15
   ...
   ```

**Usage:** These embeddings can be used for:
- Downstream classification (cancer subtyping)
- Clustering analysis
- Survival prediction
- Biomarker discovery

---

## Mathematical Formulation

### Complete Model Equations

**1. Node Encoding:**
```
H^(0) = {h_v^(0) = W_type(v) · x_v | v ∈ V}

For l = 1 to L:
    h_v^(l) = σ(AGG({W_r^(l) · h_u^(l-1) | u ∈ N_r(v), r ∈ R}))
```

Where:
- `V`: Set of all nodes
- `N_r(v)`: Neighbors of `v` via relation `r`
- `R`: Set of edge types (relations)
- `W_type(v)`: Input projection for node type
- `W_r^(l)`: Relation-specific weight matrix at layer `l`
- `AGG`: Aggregation function (attention or mean)

**2. Modality Pooling:**
```
z_mrna = POOL({W_mrna · h_v | v ∈ V_gene}, batch)
z_cnv = POOL({W_cnv · h_v | v ∈ V_gene}, batch)
z_cpg = POOL({h_v | v ∈ V_cpg}, batch)
z_mirna = POOL({h_v | v ∈ V_mirna}, batch)
```

**3. Attention Fusion (Stage B):**
```
e_m = MLP(z_m)  for m ∈ {mrna, cnv, cpg, mirna}
α_m = exp(e_m / τ) / Σ_m' exp(e_m' / τ)
h_fused = Σ_m α_m · z_m
```

**4. Feature Reconstruction:**
```
f̂_v = DEC_type(v)(h_v)
```

**5. Edge Reconstruction:**
```
ŷ_{uv} = σ(h_u^T · h_v)
```

**6. Total Objective:**
```
min_θ L_total = Σ_m λ_m ||f_m - f̂_m||²
              + λ_edge · BCE(y, ŷ)
              + λ_cons · ||h_fused - z_m||² [Stage B]
              + λ_ent · H(α) [Stage B]
```

---

## Practical Usage Guide

### Running Training

**Basic K-fold cross-validation:**
```bash
python scripts/train_level1.py --config config/default.yaml --kfold 5
```

**With custom hyperparameters:**
```bash
python scripts/train_level1.py \
    --config config/default.yaml \
    --kfold 10 \
    --batch_size 64 \
    --lr 0.001 \
    --epochs 200 \
    --device cuda \
    --seed 42
```

### Configuration File Reference

Key config paths (in `config/default.yaml`):

```yaml
# Encoder selection
model.encoder.type: 'HeteroGAT'  # or 'HeteroRGCN'

# Model dimensions
model.encoder.hidden_size: 256
model.encoder.num_layers: 3

# Training stages
training.stage_a.epochs: 100
training.stage_a.learning_rate: 0.001
training.stage_b.epochs: 100
training.stage_b.learning_rate: 0.0001

# Loss weights
losses.lambda_recon_mrna: 1.0
losses.lambda_edge: 0.5
losses.lambda_cons: 0.1
losses.lambda_ent: 0.01

# K-fold settings
kfold.n_folds: 5
kfold.stratified: false
```

### Hyperparameter Tuning Recommendations

**Encoder Type:**
- Use `HeteroGAT` for datasets with informative graph structure
- Use `HeteroRGCN` for simpler relational modeling or smaller datasets

**Hidden Size:**
- 128: Faster, less expressive (small datasets <100 patients)
- 256: Balanced (default, recommended)
- 512: More expressive (large datasets >500 patients, requires more memory)

**Learning Rates:**
- Stage A: 0.001 (default), reduce to 0.0005 if loss diverges
- Stage B: 0.0001 (10× smaller than Stage A is critical)

**Loss Weights:**
- `lambda_edge`: 0.5 (default), increase to 1.0 for sparse graphs
- `lambda_cons`: 0.1 (default), decrease to 0.05 if attention becomes too uniform
- `lambda_ent`: 0.01 (default), increase to 0.05 to encourage more diverse attention

**Batch Size:**
- GPU memory limited: Use 16 or 32
- Large GPU (16GB+): Can use 64 or 128
- CPU: Use 8 or 16

### Monitoring Training

**Key metrics to watch:**

1. **Reconstruction MSE**: Should decrease steadily in Stage A
   - Target: <0.1 for normalized data
   - If stuck, check data normalization

2. **Edge AUROC**: Should reach >0.7 for meaningful edge prediction
   - <0.6: Graph structure may not be informative
   - >0.9: Very strong graph signal

3. **Consistency Loss**: Should be low in Stage B
   - Target: <0.5 after Stage B training
   - High values: Modality embeddings are diverging from fusion

4. **Attention Weights**: Check for balance across modalities
   - Balanced: All modalities between 0.15-0.35
   - Collapsed: One modality >0.8 (increase λ_ent)

### Troubleshooting

**Issue: Training loss not decreasing**
- Check data normalization (mean≈0, std≈1)
- Reduce learning rate by 10×
- Increase gradient clipping threshold

**Issue: NaN or Inf losses**
- Reduce learning rate
- Check for missing values in data
- Enable gradient clipping (default: 1.0)

**Issue: Attention collapse (one modality dominates)**
- Increase `lambda_ent` from 0.01 to 0.05
- Check modality weighting (set `losses.modality_weights` manually)

**Issue: Poor edge reconstruction**
- Increase `lambda_edge` from 0.5 to 1.0
- Check edge data quality (remove duplicate edges)
- Increase `neg_sampling_ratio` from 5 to 10

**Issue: Out of memory**
- Reduce `batch_size` (try 16 or 8)
- Reduce `hidden_size` (try 128)
- Reduce `num_layers` (try 2)
- Enable `mixed_precision: true`

---

## Performance Benchmarks

**Typical Training Times** (on NVIDIA A100):

| Dataset Size | Patients | Genes | CpGs | miRNAs | Stage A | Stage B | Per Fold |
|--------------|----------|-------|------|--------|---------|---------|----------|
| Small        | 50       | 5K    | 10K  | 500    | 5 min   | 5 min   | 10 min   |
| Medium       | 200      | 10K   | 50K  | 1K     | 15 min  | 15 min  | 30 min   |
| Large        | 500      | 20K   | 100K | 2K     | 45 min  | 45 min  | 90 min   |

**Memory Requirements:**

| Configuration | GPU Memory | Peak Memory |
|---------------|------------|-------------|
| Batch=16, H=128 | 4 GB    | 6 GB        |
| Batch=32, H=256 | 8 GB    | 12 GB       |
| Batch=64, H=512 | 16 GB   | 24 GB       |

---

## Summary

The Level 1 model provides a comprehensive framework for multi-modal genomics integration:

✅ **Heterogeneous Graph Structure**: Naturally represents biological relationships
✅ **Modality-Specific Processing**: Separate pathways for each data type
✅ **Learned Attention Fusion**: Patient-specific modality importance
✅ **Self-Supervised Learning**: No labels required for training
✅ **Robust Evaluation**: K-fold cross-validation with multiple metrics
✅ **Interpretable Outputs**: Attention weights reveal modality importance

### Key Innovations

1. **Dual-channel gene nodes** (mRNA + CNV) with separate decoding
2. **Two-stage training** (reconstruction → fusion)
3. **Attention-based modality fusion** with entropy regularization
4. **Consistency loss** to maintain modality coherence

### Limitations

- Single-patient graphs (no inter-patient relationships)
- Fixed graph structure (no learned edges)
- Requires all 4 modalities for each patient
- No hierarchical modeling (Level 2 not implemented)
- No transfer learning across cancer types

### Future Extensions (Level 2)

- **Population graph**: Model relationships across patients
- **Hierarchical modeling**: genes → pathways → biological processes → cancer types
- **Multi-task learning**: Joint training on survival, classification, drug response
- **Transfer learning**: Pre-train on large cohort, fine-tune on target dataset
- **Learned graph structure**: Discover new biological relationships

### Related Documentation

- **User Guide**: See [README.md](../README.md) for installation and usage
- **Development Guide**: See [CLAUDE.md](../CLAUDE.md) for code contribution guidelines
- **Configuration**: See [config/default.yaml](../config/default.yaml) for all parameters

---

**Document Version**: 1.0
**Last Updated**: 2024-10-23
**Corresponding Code Version**: Level 1 (main branch)
