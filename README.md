# Level‑1 Multi‑Omics Heterogeneous GNN (Unsupervised)

> **Goal:** For each patient, build an intra‑patient heterogeneous graph (gene / CpG / miRNA), learn **per‑modality embeddings** (mRNA, CNV, DNAmeth, miRNA), and fuse them with **modality‑level attention** into a **patient embedding** `h_i`. Train **unsupervised** with reconstruction + regularizers. **No modality dropout.** Inputs are assumed **normalized + imputed**.

This README gives you everything you need: data schemas, project layout, math, config guide, commands, exports, evaluation ideas, and how to extend to new modalities.

---

## 0) Tech stack & conventions

- **Python** 3.10+
- **PyTorch** ≥ 2.1
- **PyTorch Geometric (PyG)** ≥ 2.4 (preferred). (If you must use DGL later, swap the encoder and loaders.)
- **Config‑driven:** YAML (`config/*.yaml`) + simple CLI.
- **Reproducibility:** global seeds are set; CuDNN deterministic where possible.
- **Style:** Shapes in brackets like `[B, D]`. “Patient” ≈ a single per‑patient graph in a batch.

---

## 1) What the model learns (big picture)

For each patient _i_:

- Node types: **gene**, **cpg**, **mirna**.
- Edges (inside patient graph):
  - `('cpg','maps_to','gene')`
  - `('mirna','targets','gene')`
  - Optional: `('gene','ppi','gene')` (enable via config `data.add_ppi`).
- Node features (per patient column from TSVs):
  - **gene**: 2‑channel features `[expr, cnv]` _(mRNA expression, CNV)_.
  - **cpg**: methylation value.
  - **mirna**: miRNA expression value.
- Encoder: **Hetero‑GNN** (GATv2 by default) → node embeddings per type.
- **Per‑node‑type pooling** → graph‑level embeddings per node type.
- **Gene heads:** a small two‑head MLP splits the pooled **gene** vector into **`z^(mRNA)`** and **`z^(CNV)`**.
- **Modality attention:** scores each modality vector, softmax → attention weights **α**; fuse → patient embedding **`h_i`**.
- **Unsupervised losses:** reconstruct node features **and** edges; add regularizers (consistency + attention entropy).

---

## 2) Data layout & schemas

Put files under `data/`:

```
data/
  features/
    genes_expr.tsv   # rows: gene IDs,    cols: patient IDs (float)
    genes_cnv.tsv    # rows: gene IDs,    cols: patient IDs (float)
    cpgs.tsv         # rows: CpG IDs,     cols: patient IDs (float)
    mirnas.tsv       # rows: miRNA IDs,   cols: patient IDs (float)
    samples.txt      # one patient ID per line (must match the TSV column names)
  edges/
    gene_cpg.csv     # columns: cpg_id,gene_id[,weight]
    gene_mirna.csv   # columns: mirna_id,gene_id[,weight]
    gene_gene.csv    # OPTIONAL PPI/pathway: gene_id_src,gene_id_dst[,weight]
```

### 2.1 TSV format (features)

- **Header row** = patient IDs (`P001`, `P002`, …).
- **Col 0** = feature ID (`id` or similar).
- **Values** = **already normalized + imputed** floats.
- **All four TSVs share the identical patient columns in the same order.**
- **`genes_expr.tsv` and `genes_cnv.tsv` must have identical gene rows** (same IDs, same order).

**Tiny example (`genes_expr.tsv`):**

```
id      P1      P2      P3
ENSG1   0.12    -0.01   0.77
ENSG2   0.30     0.45  -0.20
...
```

**Tiny example (`cpgs.tsv`):**

```
id      P1      P2      P3
cg0001  0.85    0.88    0.81
cg0002  0.61    0.59    0.63
...
```

### 2.2 CSV format (edges)

- **Gene–CpG (`gene_cpg.csv`)**: `cpg_id,gene_id[,weight]`
- **miRNA–Gene (`gene_mirna.csv`)**: `mirna_id,gene_id[,weight]`
- **Gene–Gene (`gene_gene.csv`)** (optional): `gene_id_src,gene_id_dst[,weight]`
- If `weight` is omitted, **1.0** is used.
- IDs must exist in corresponding TSV rows.

**Example (`gene_mirna.csv`):**

```
mirna_id,gene_id,weight
hsa-miR-21,ENSG1,1.0
hsa-miR-34a,ENSG7,0.6
```

---

## 3) Project structure

```
project/
  config/
    default.yaml       # general defaults
    train.yaml         # main experiment config
  data/
    features/          # TSVs + samples.txt
    edges/             # CSVs
  src/
    dataio/            # TSV/CSV loaders, hetero-graph builder, dataset
    models/            # hetero encoder, pooling, attention
    losses/            # feature/edge recon, consistency, entropy, weighting
    train/             # trainer, (future) scheduler
    utils/             # seed, logging, ckpt helpers
  scripts/
    train_level1.py
    eval_level1.py     # minimal placeholder
  outputs/
    checkpoints/
    logs/
    tensors/
  requirements.txt
  README.md
```

---

## 4) Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# IMPORTANT: PyG has specific wheels per Torch/CUDA.
# Follow: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```

**CUDA note:** Make sure your PyTorch and CUDA versions match the PyG wheel you install.

---

## 5) Configuration (line‑by‑line guide)

Open `config/train.yaml`. Important keys:

```yaml
seed: 1337 # global seed
device: auto # "cuda", "cpu", or "auto"

data:
  features_dir: ./data/features
  edges_dir: ./data/edges
  genes_expr_tsv: genes_expr.tsv
  genes_cnv_tsv: genes_cnv.tsv
  cpgs_tsv: cpgs.tsv
  mirnas_tsv: mirnas.tsv
  samples_txt: samples.txt
  gene_cpg_csv: gene_cpg.csv
  gene_mirna_csv: gene_mirna.csv
  gene_gene_csv: gene_gene.csv
  add_ppi: false # set true to load gene-gene edges

model:
  hidden_dim: 64 # node hidden size inside the encoder
  out_dim: 64 # final per-node embedding size from encoder
  num_layers: 2
  dropout: 0.1
  layernorm: true
  conv: gatv2 # gatv2 | graph | sage
  heads: 2 # for gatv2

pooling:
  type: mean # mean | attn (attention pooling per node type)
  attn_hidden: 64

modalities:
  dim: 64 # size of each modality vector z^(m)
  names: [mRNA, CNV, DNAmeth, miRNA]

loss:
  recon_eta: 1.0 # scales edge losses inside L_recon
  alpha_consistency: 1.0
  beta_entropy: 0.05
  gamma_contractive: 0.0 # (not used by default)
  lambda: # per-modality weights (will be normalized)
    mRNA: 1.0
    CNV: 1.0
    DNAmeth: 1.0
    miRNA: 1.0

train:
  stage: both # runs Stage A then Stage B
  batch_size: 8
  max_epochs_stageA: 2 # set higher for real runs
  max_epochs_stageB: 2
  lr: 0.001
  weight_decay: 0.0
  patience: 3 # (for future early stopping)
  num_workers: 0
  val_split: 0.2
  shuffle_patients: true

export:
  out_dir: ./outputs
  save_best: true
  save_attention_csv: true
```

**Switch to attention pooling** per node type:

```yaml
pooling:
  type: attn
  attn_hidden: 64
```

---

## 6) How to run

```bash
python scripts/train_level1.py --config config/train.yaml
```

The repo includes **tiny toy data** so you can do a smoke test. Replace the TSV/CSV files with your real cohort when ready.

---

## 7) Deep dive: shapes and flow

### 7.1 Per‑patient graph (PyG `HeteroData`)

- `data["gene"].x` → shape `[G, 2]` (columns: `expr`, `cnv` for that patient)
- `data["cpg"].x` → `[C, 1]`
- `data["mirna"].x` → `[M, 1]`
- Edges per relation have `edge_index` `[2, E_r]` and optional `edge_weight` `[E_r]`.

### 7.2 Encoder

- HeteroGNN (GATv2 default) with relation‑specific message passing.
- Output: node embeddings per type, e.g. `gene: [G, D_node]`, `cpg: [C, D_node]`, `mirna: [M, D_node]`
  where `D_node = model.out_dim`.

### 7.3 Pooling → modality vectors

- Pool **per node type** across nodes in each graph (batch‑aware).
- **Gene pooled vector** → two small heads → **`z^(mRNA)`** and **`z^(CNV)`**, each `[B, D_mod]`.
- CpG pooled → linear → **`z^(DNAmeth)`** `[B, D_mod]`.
- miRNA pooled → linear → **`z^(miRNA)`** `[B, D_mod]`.
- `D_mod = modalities.dim`.

### 7.4 Modality attention fusion

- Score each modality vector with a shared small MLP → scalar scores `s_i^(m)`.
- Softmax over modalities → weights `α_i^(m)` (sum to 1 across modalities).
- **Fused patient embedding:** `h_i = Σ_m α_i^(m) z_i^(m)`.

---

## 8) Losses (Level‑1, unsupervised; no modality dropout)

Let `R(m)` be edge relations tied to modality `m`:

- `DNAmeth ↔ ('cpg','maps_to','gene')`
- `miRNA ↔ ('mirna','targets','gene')`
- `mRNA`/`CNV` have only the **feature** recon terms (by design here).

### 8.1 Feature reconstruction (node‑type aware)

- **Gene:** **two decoders** predict **mRNA** and **CNV** channels separately → MSE.
- **CpG / miRNA:** one decoder each → MSE.

### 8.2 Edge reconstruction (per relation)

- Binary link prediction with negative sampling; BCE on pos/neg logits.

### 8.3 Consistency

Encourage fused `h_i` to stay close to each modality vector:

```
L_cons = (1/M) * Σ_m || h_i − z_i^(m) ||_2^2   (average over modalities)
```

### 8.4 Attention entropy

Discourage overly peaky attention:

```
H(α_i) = − Σ_m α_i^(m) log(α_i^(m) + ε)
L_ent  = − (1/N) Σ_i H(α_i)      # small negative‑entropy penalty
```

### 8.5 Combine

```
L_recon = Σ_m λ_m [ L_feat^(m) + η Σ_{r∈R(m)} L_edge^(r) ]
L_total = L_recon + α L_cons + β L_ent + γ L_contractive(optional)
```

All scalars (`λ_m`, `η`, `α`, `β`, `γ`) are set in YAML. `λ_m` is normalized in code.

---

## 9) Training protocol

- **Stage A (pretrain):** train encoder + decoders with **reconstruction losses only** until plateau.
- **Stage B (fusion):** continue training with **attention + consistency + entropy** terms enabled (full loss).

The provided trainer runs **both** stages back‑to‑back when `train.stage: both` (default).

---

## 10) Exports & logging

During/after training:

- **Logs:** `outputs/logs/train_log.csv` with per‑epoch `L_total`, `L_recon`, `L_cons`, `L_ent` for train/val.
- **Embeddings:** (exported at the end of Stage B)
  - `outputs/tensors/patient_embeddings.pt` — tensor `[N, D_mod]` of all `h_i`.
  - `outputs/tensors/modality_embeddings.pt` — dict `{mRNA,CNV,DNAmeth,miRNA: [N, D_mod]}`.
  - `outputs/tensors/attention_weights.csv` — rows = patients (in batch order), cols = `[mRNA,CNV,DNAmeth,miRNA]`.
- **Checkpoints:** (you can add saving the best epoch if needed; helper functions are in `src/utils/ckpt.py`).

**Tip:** When you move to a real dataset, bump `max_epochs_stageA/B`, tune `hidden_dim`, `out_dim`, and `modalities.dim`.

---

## 11) Minimal evaluation (ideas to extend)

The placeholder `scripts/eval_level1.py` can be extended to report:

- **Feature recon MSE** per node type on a held‑out split.
- **AUROC/AUPRC** for each edge relation (by scoring positive vs. sampled negative pairs from the encoder).
- **Attention stats:** mean/variance of `α` per modality; cohort‑level histograms.
- **Ablations:** turn off a modality (set its `λ_m=0`) and observe changes in recon or attention.

---

## 12) Preparing your own data (checklist)

- ✔ Normalize + impute all feature matrices first.
- ✔ Ensure **same patients** across all TSVs (**identical order** of columns).
- ✔ Ensure **same genes and order** between `genes_expr.tsv` and `genes_cnv.tsv`.
- ✔ Check IDs in edge CSVs **exist** in the TSV row IDs.
- ✔ If adding `gene_gene.csv`, set `data.add_ppi: true` in YAML.
- ✔ Large cohorts: increase `batch_size`, use GPU, and consider attention pooling per node type.

**Quick sanity script** (pseudocode):

```python
# Load TSVs with pandas and assert row/column alignment
assert list(genes_expr.index) == list(genes_cnv.index)
for tsv in [genes_expr, cpgs, mirnas]:
    assert list(tsv.columns) == patient_list_from_samples_txt
```

---

## 13) Changing architecture & hyper‑params

- **Conv type:** set `model.conv: graph` (GraphConv) or `sage` (SAGEConv).
- **Deeper encoder:** increase `model.num_layers`.
- **Node dims:** `model.hidden_dim`, `model.out_dim`.
- **Modality dims:** `modalities.dim` (affects `z^(m)` and `h_i`).
- **Pooling:** `pooling.type: mean` → fastest; `attn` → a bit stronger, slightly slower.
- **Loss balance:** tune `loss.lambda.*`, `loss.recon_eta`, `loss.alpha_consistency`, `loss.beta_entropy`.

---

## 14) Extending to a new modality (example: Proteomics)

1. Add a TSV `proteins.tsv` (rows: protein IDs; cols: patients).
2. Add node type **protein** in the graph builder (optional); or treat as another channel on **gene** and add a third head.
3. Create a decoder for protein features in `FeatureDecoders`.
4. Update modality pooling to generate `z^(Protein)` and wire it into attention.
5. Add `lambda.Protein` in YAML.

This project already shows how to split a multi‑channel node (`gene`) into two modalities (mRNA/CNV), which you can generalize.

---

## 15) Performance tips

- Start with **mean pooling**; switch to `pooling.attn` later.
- Increase **batch size** and **hidden_dim** gradually.
- If edge CSVs are huge, pre‑filter or weight sparsely.
- Use **mixed precision** (add `torch.cuda.amp`) for large runs.
- To stabilize attention early, lower `beta_entropy` (e.g., 0.01) or warm up Stage B after more Stage A epochs.

---

## 16) Troubleshooting

- **`AssertionError: … TSV columns must match`:** patient columns differ; reorder & rename columns to be identical.
- **`genes_expr` vs `genes_cnv` mismatch:** row sets/order must be the same.
- **PyG install errors:** your Torch/CUDA/PyG versions don’t match; reinstall using the official commands.
- **Empty relations:** if an edge CSV is empty or IDs don’t map, the code inserts a tiny dummy edge to keep shapes valid; fix the CSV.
- **All attention on one modality:** lower `loss.beta_entropy` a bit or increase `pooling.attn_hidden`; also check normalization.

---

## 17) Reproducibility

- Global seed in YAML (`seed:`).
- CuDNN deterministic flags set.
- For true bit‑wise repeatability across GPUs and backends, keep versions pinned.

---

## 18) License & citation

Use freely for research/education. If you publish, please cite your own dataset/resources accordingly.

---

## 19) Commands cheat‑sheet

```bash
# Train with defaults
python scripts/train_level1.py --config config/train.yaml

# Edit config and run again
vim config/train.yaml

# Inspect logs
column -t -s, outputs/logs/train_log.csv | less -S

# After training: exported tensors
ls outputs/tensors/
```

---

## 20) What’s inside `outputs/tensors/`

- `patient_embeddings.pt`: PyTorch tensor `[N, D_mod]` (fused `h_i`).
- `modality_embeddings.pt`: `dict[str, Tensor]` with keys `{mRNA,CNV,DNAmeth,miRNA}`, each `[N, D_mod]`.
- `attention_weights.csv`: CSV with columns `[mRNA,CNV,DNAmeth,miRNA]` and `N` rows (batch order).

---

### Final notes

- This Level‑1 repo stops at **unsupervised** embeddings. Downstream tasks (clustering, survival, prediction) can be added later using `h_i`.
- Level‑2 (e.g., cross‑patient graph or supervised heads) is **out of scope here** by design.

Happy training! 🚀
