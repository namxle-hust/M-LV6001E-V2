# Level 1: Multi‑Modal Heterogeneous GNN

## TL;DR
- Stage A pretrains the encoder and projection heads with reconstruction + edge + projection losses, producing modality‑aware, structure‑aware embeddings.
- Stage B fine‑tunes the same weights with attention‑based fusion, adding consistency and entropy losses to learn patient‑specific modality weighting.
- Projection heads isolate mRNA vs CNV from shared gene embeddings; attention fuses mRNA, CNV, CpG, and miRNA per patient.

---

## Data & Graph
- Nodes: genes, CpGs, miRNAs.
- Edges: CpG→gene (maps_to), miRNA→gene (targets), (optional) gene↔gene (PPI). Reverse edges may be present for message passing.
- Features:
  - Gene: 2 channels (mRNA expression, CNV) per gene node.
  - CpG: methylation values.
  - miRNA: expression values.

---

## Model Overview
- Encoder: heterogeneous GNN (e.g., HeteroGAT/RGCN) → per‑node hidden vectors h ∈ R^d.
- Modality pooling: maps node embeddings to patient‑level modality embeddings z_m ∈ R^{B×d}.
  - Gene→mRNA: z_mrna via a learned projection head + pooling.
  - Gene→CNV: z_cnv via a separate learned projection head + pooling.
  - CpG, miRNA: pooled directly (mean/sum/max or attention pooling).
- Fusion:
  - Stage A: simple mean of modality embeddings.
  - Stage B: learned attention over modalities to produce a fused patient embedding.

---

## Projection Heads (Gene → mRNA/CNV)
- Two independent linear maps W_mrna, W_cnv ∈ R^{d×d} applied to gene embeddings:
  - z_g^mrna = W_mrna · h_g,    z_g^cnv = W_cnv · h_g
- Pool per patient (e.g., mean) to get patient‑level z_mrna, z_cnv.
- Why: isolate modality‑specific directions from a shared gene representation; each head learns to emphasize features predictive of its modality.

---

## Stage A: Pretraining (Reconstruction + Projections)

### Forward
1) Encode nodes with the hetero‑GNN → node embeddings h.
2) Apply projection heads on genes (mRNA/CNV), then pool; pool CpG/miRNA → z_{mrna,cnv,cpg,mirna}.
3) Compute a simple fused mean: h_fused = mean({z_m}).

### Losses
1) Feature reconstruction (per modality):
   - Reconstruct original features from node embeddings via decoders.
   - L_recon = Σ_m λ_recon_m · MSE(f̂_m, f_m)

2) Edge reconstruction (graph structure):
   - Inner‑product/MLP decoder with negative sampling across relations.
   - Binary cross‑entropy on positives/negatives → L_edge.

3) Projection reconstruction (mRNA & CNV only):
   - Targets per patient b (mean over genes G):
     - μ_mrna^{(b)} = (1/G) Σ_g x_mrna^{(b,g)}
     - μ_cnv^{(b)}  = (1/G) Σ_g x_cnv^{(b,g)}
   - Small decoders map z_mrna, z_cnv → scalar predictions p_mrna, p_cnv.
   - L_proj = ½ [ MSE(p_mrna, μ_mrna) + MSE(p_cnv, μ_cnv) ].

4) Projection diversity regularization:
   - Encourage z_mrna ≠ z_cnv via cosine similarity penalty:
     - L_div = E_b [ |cos(z_mrna^{(b)}, z_cnv^{(b)})| ].

5) Stage A total objective:
   - L_A = L_recon + λ_edge · L_edge + λ_projection · L_proj + λ_proj_reg · L_div

Notes
- Only mRNA/CNV participate in L_proj and L_div; CpG/miRNA are unaffected by projection loss.
- Gradients flow into encoder and projection heads, improving modality separation before attention.

---

## Stage B: Fusion (Attention + Consistency + Entropy)

### Forward
- Reuse Stage‑A‑trained weights; re‑encode the batch graphs.
- Compute modality embeddings z_m again; apply attention to fuse:
  - Scores: s_m = MLP(z_m)
  - Weights: α_m = softmax(s_m / T)
  - Fused: h_fused = Σ_m α_m · z_m

### Losses
1) Keep reconstruction and edge terms:
   - L_recon (feature), λ_edge · L_edge (structure) remain active to preserve content/geometry.

2) Consistency (align fused and modalities):
   - With normalization, default distance is L2:
     - L_cons = (1/M) Σ_m || normalize(h_fused) − normalize(z_m) ||_2
   - Role: keeps h_fused faithful to all modality embeddings; gradients backprop through attention → z_m → projections → encoder.

3) Entropy regularization (healthy attention):
   - Per patient entropy: H(α) = −Σ_m α_m log α_m
   - Loss maximizes entropy by minimizing negative entropy:
     - L_ent = −E_b [ H(α) ]
   - Role: discourages early single‑modality collapse; tuned via λ_ent and temperature T.

4) Stage B total objective:
   - L_B = L_recon + λ_edge · L_edge + λ_cons · L_cons + λ_ent · L_ent

Notes
- Projection‑specific losses (L_proj, L_div) are disabled in Stage B to avoid conflicting proxy objectives; projections still update via L_cons gradients.

---

## Why Two Stages (Not One)
- Stability: Stage A warms up modality‑/structure‑aware embeddings before attention.
- Reduced collapse: Pretrained projections make “make all z_m equal” less attractive.
- Simpler tuning: Separate λ schedules (projection vs fusion) without complex annealing.
- Clear roles: Stage A learns representations; Stage B learns patient‑specific fusion.

---

## Handling Modality Imbalance (e.g., CpG 450k vs miRNA 1k)
- Mean pooling prevents raw node‑count dominance (avoid sum pooling).
- Consistency uses normalized vectors, adding scale invariance.
- Entropy regularization discourages always‑peaky attention on a single modality.
- Additional practices: robust per‑modality normalization; tune λs; optional modality dropout; adjust temperature T.

---

## Gradient Flow Summary
- Stage A: L_recon → decoders → nodes → encoder; L_edge → nodes → encoder; L_proj/L_div → z_mrna/z_cnv → projections → encoder.
- Stage B: L_cons → attention → z_m → projections → encoder; L_ent → attention; L_recon/L_edge continue to update encoder/decoders.

---

## Key Hyperparameters (config)
- λ_edge: structure preservation.
- Stage A: λ_projection (projection reconstruction), λ_proj_reg (diversity).
- Stage B: λ_cons (consistency), λ_ent (entropy), attention temperature T.
- Pooling type: mean or attention at pooling stage.

Tuning Tips
- If attention is too peaky: increase λ_ent or raise T.
- If attention is too flat: lower λ_ent or lower T; ensure λ_cons is sufficient.
- If projections look similar: raise λ_proj_reg (Stage A) or add a small diversity term in Stage B.
- If structure drifts: increase λ_edge in Stage B (or avoid setting it to 0).

---

## Outputs & Monitoring
- Per‑modality reconstruction MSEs and edge AUROC (sanity checks).
- Stage A only: projection loss and projection_reg trends (pretraining health).
- Stage B: consistency, entropy; attention weights per patient (exported), fused embeddings.

---

## Frequently Asked Questions
1) Do projection heads train in Stage B?
   - Yes, indirectly via L_cons. Projection‑specific losses are Stage A only.

2) Why compute a “fused” vector in Stage A?
   - API consistency and diagnostics; it’s a simple mean and not used by Stage A losses.

3) Is this transfer learning?
   - It’s a pretrain‑then‑finetune curriculum on the same data: Stage A initializes; Stage B fine‑tunes under fusion losses.

4) Why keep edge reconstruction in Stage B?
   - Acts as a structural anchor; reduces drift and collapse during fusion training.

---

## Mathematical Summary
- Stage A:
  - L_A = Σ_m λ_recon_m · MSE(f̂_m, f_m) + λ_edge · L_edge
           + λ_projection · ½ [ MSE(p_mrna, μ_mrna) + MSE(p_cnv, μ_cnv) ]
           + λ_proj_reg · E_b [ |cos(z_mrna^{(b)}, z_cnv^{(b)})| ]

- Stage B:
  - h_fused = Σ_m α_m z_m,   α = softmax(MLP(z)/T)
  - L_cons = (1/M) Σ_m || normalize(h_fused) − normalize(z_m) ||_2
  - L_ent = −E_b [ Σ_m α_m log α_m ]
  - L_B = Σ_m λ_recon_m · MSE(f̂_m, f_m) + λ_edge · L_edge + λ_cons · L_cons + λ_ent · L_ent

---

## Takeaways
- Stage A makes projections and encoder modality‑ and structure‑aware.
- Stage B learns patient‑specific fusion while preserving content and structure.
- The two‑stage design improves stability, convergence speed, and fused embedding quality compared to training everything at once.

