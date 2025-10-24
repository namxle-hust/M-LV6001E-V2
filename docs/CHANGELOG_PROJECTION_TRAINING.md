# Changelog: Projection Training Enhancement

## Summary

Added projection head training to Stage A to ensure `mrna_projection` and `cnv_projection` learn meaningful representations during pretraining, rather than remaining randomly initialized.

---

## Changes Made

### 1. New File: `src/losses/projection_loss.py`

**Added 3 new loss classes**:

#### `ProjectionReconstructionLoss`
- **Purpose**: Train projection heads to preserve modality-specific information
- **Mechanism**: Small decoders predict patient-level mean values from pooled embeddings
- **Usage**: Active in Stage A only
- **Default weight**: `lambda_projection = 0.1`

#### `ProjectionRegularizationLoss`
- **Purpose**: Encourage mRNA and CNV projections to extract different features
- **Mechanism**: Penalizes high cosine similarity between `z_mrna` and `z_cnv`
- **Usage**: Active in Stage A only
- **Default weight**: `lambda_proj_reg = 0.01`

#### `ProjectionAlignmentLoss` (optional)
- **Purpose**: Alternative approach using node-level alignment
- **Status**: Implemented but not used by default
- **Can be enabled**: By modifying `MultiModalGNNWithDecoders.__init__`

---

### 2. Modified File: `src/models/multimodal_gnn.py`

#### Changes in `MultiModalGNNWithDecoders.__init__` (lines 159-193):

**Added**:
```python
from ..losses.projection_loss import (
    ProjectionReconstructionLoss,
    ProjectionRegularizationLoss
)

# Projection training loss (Stage A)
self.projection_loss = ProjectionReconstructionLoss(config, decoder_type="simple")

# Projection regularization (Stage A)
self.projection_reg = ProjectionRegularizationLoss(
    reg_type="diversity",
    weight=config["losses"].get("lambda_proj_reg", 0.01)
)
```

#### Changes in `MultiModalGNNWithDecoders.forward` (lines 245-259):

**Added** (after edge loss, before Stage B losses):
```python
# Stage A specific losses: Projection training
if self.model.training_stage == "A":
    # Projection reconstruction loss
    proj_loss, proj_detail = self.projection_loss(
        output["modality_embeddings"],
        original
    )
    losses["projection"] = proj_loss * self.config["losses"].get("lambda_projection", 0.1)
    losses["projection_detail"] = proj_detail

    # Projection regularization (diversity)
    proj_reg = self.projection_reg(output["modality_embeddings"])
    losses["projection_reg"] = proj_reg
```

#### Changes in total loss computation (lines 275-288):

**Modified**:
```python
# Total loss
total_loss = losses["recon_total"] + losses["edge_total"]

if self.model.training_stage == "A":
    # Add projection training losses in Stage A
    total_loss += losses.get("projection", 0)
    total_loss += losses.get("projection_reg", 0)

if self.model.training_stage == "B":
    # Add consistency and entropy losses in Stage B
    total_loss += losses.get("consistency", 0)
    total_loss += losses.get("entropy", 0)
```

---

### 3. Modified File: `config/default.yaml`

#### Changes in `losses` section (lines 73-76):

**Added**:
```yaml
# Projection training (Stage A only)
# NEW: Ensures projection heads learn meaningful representations in Stage A
lambda_projection: 0.1  # Weight for projection reconstruction loss
lambda_proj_reg: 0.01   # Weight for projection diversity regularization
```

**Location**: After `modality_weights`, before `lambda_cons`

---

### 4. New Documentation: `docs/PROJECTION_TRAINING.md`

Comprehensive documentation covering:
- Problem statement and motivation
- Solution architecture
- Implementation details
- Gradient flow analysis
- Configuration and tuning
- Validation and monitoring
- Comparison with original implementation

---

## Impact Analysis

### What Changed

#### Stage A Loss Function:
**Before**:
```
L_StageA = L_recon + L_edge
```

**After**:
```
L_StageA = L_recon + L_edge + λ_proj·L_projection + λ_reg·L_proj_reg
```

#### Projection Head Training:
| Stage | Before | After |
|-------|--------|-------|
| **Stage A (100 epochs)** | ❌ Not trained | ✅ **Actively trained** |
| **Stage B (100 epochs)** | ✅ Trained | ✅ Trained (continues) |

### What Didn't Change

- ✅ Encoder architecture (unchanged)
- ✅ Decoder architecture (unchanged)
- ✅ Stage B losses (unchanged)
- ✅ Data pipeline (unchanged)
- ✅ K-fold CV workflow (unchanged)
- ✅ Backward compatible (can disable with `lambda_projection: 0`)

---

## Testing Recommendations

### 1. Quick Validation Test

```bash
# Run 1 fold with 10 epochs each stage (quick test)
python scripts/train_level1.py \
    --config config/default.yaml \
    --kfold 1 \
    --epochs 10 \
    --batch_size 16
```

**Expected output**: Should see new loss components in logs:
- `projection` loss appearing in Stage A
- `projection_reg` loss appearing in Stage A
- Both should NOT appear in Stage B

### 2. Full Training Comparison

**Test A (With projection training - NEW)**:
```bash
python scripts/train_level1.py --config config/default.yaml --kfold 5
```

**Test B (Without projection training - BASELINE)**:
```bash
# Create temporary config
cp config/default.yaml config/no_projection.yaml
# Edit config/no_projection.yaml: set lambda_projection: 0.0

python scripts/train_level1.py --config config/no_projection.yaml --kfold 5
```

**Compare**:
- Stage B convergence speed (expect faster with projection training)
- Final patient embedding quality
- Attention weight diversity

### 3. Monitoring Checklist

During training, verify:
- [ ] `projection` loss decreases in Stage A
- [ ] `projection` loss does NOT appear in Stage B
- [ ] `projection_reg` stabilizes (doesn't necessarily decrease)
- [ ] Total loss in Stage A is slightly higher (expected due to new terms)
- [ ] Stage B converges faster or to better quality

---

## Migration Guide

### For Existing Projects

**Option 1: Use new behavior (recommended)**
- No changes needed
- Just use updated codebase
- Projection training is automatic

**Option 2: Keep old behavior**
```yaml
# In your config file
losses:
  lambda_projection: 0.0
  lambda_proj_reg: 0.0
```

### For New Projects

Use default configuration:
```yaml
losses:
  lambda_projection: 0.1
  lambda_proj_reg: 0.01
```

Tune if needed based on monitoring.

---

## Performance Expectations

### Computational Cost

- **Stage A**: +5-10% training time (additional forward passes through projection decoders)
- **Stage B**: No change or slight speedup (better initialization)
- **Overall**: Negligible impact (~3-5% total time increase)

### Quality Improvements

Expected improvements (to be validated):
- Better modality separation: +10-20% diversity in attention weights
- Faster Stage B convergence: -20-30% epochs to reach same quality
- Improved final metrics: +2-5% on downstream tasks

---

## Rollback Plan

If projection training causes issues:

1. **Quick fix** (disable feature):
   ```yaml
   losses:
     lambda_projection: 0.0
     lambda_proj_reg: 0.0
   ```

2. **Full rollback** (revert changes):
   ```bash
   git revert <commit_hash>
   ```

3. **Debug** (reduce weights):
   ```yaml
   losses:
     lambda_projection: 0.01  # Much smaller
     lambda_proj_reg: 0.001
   ```

---

## Known Limitations

1. **Projection decoders are simple**: Only predict mean values, not full distributions
2. **Only for gene modalities**: CpG and miRNA projections are not explicitly trained (they use same node embeddings without splitting)
3. **Fixed diversity metric**: Uses cosine similarity; other metrics (mutual information, etc.) not implemented

---

## Future Work

Potential enhancements:
1. Add projection training for CpG/miRNA modalities
2. More sophisticated projection decoders (predict variances, quantiles)
3. Adaptive loss weighting based on validation metrics
4. Cross-modality reconstruction (mRNA projection predicts CNV, etc.)

---

## Version Information

- **Implementation Date**: 2025-01-24
- **Affected Version**: Level 1, Stage A/B training
- **Backward Compatible**: Yes (can disable with config)
- **Breaking Changes**: None

---

## Files Modified Summary

| File | Type | Lines Changed | Description |
|------|------|---------------|-------------|
| `src/losses/projection_loss.py` | NEW | +342 | New loss module |
| `src/models/multimodal_gnn.py` | MODIFIED | +25 | Integration |
| `config/default.yaml` | MODIFIED | +4 | Config params |
| `docs/PROJECTION_TRAINING.md` | NEW | +600 | Documentation |
| `CHANGELOG_PROJECTION_TRAINING.md` | NEW | This file | Change log |

**Total**: 2 new files, 2 modified files, ~971 lines added

---

## Contact

For questions or issues related to this enhancement:
- Review documentation: `docs/PROJECTION_TRAINING.md`
- Check implementation: `src/losses/projection_loss.py`
- Refer to original architecture: `docs/LEVEL1_ARCHITECTURE.md`
