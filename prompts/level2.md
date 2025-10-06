## Prompt: “Generate a simple Level-2 clustering + survival verification script”

### Goal

Create a single Python 3.10+ script that uses the Level-1 patient embeddings to cluster patients and evaluates the clustering with the concordance index (C-index) using patient survival data.

### Inputs & expected files

--embeddings: path to outputs/tensors/patient_embeddings.pt (PyTorch tensor, shape [N, d]; order aligns with patient IDs below).

--patient-ids: path to data/level1/samples.txt (one patient ID per line, same order as rows in the embeddings tensor).

--meta: path to data/level2/patient_meta.tsv with columns:

patient_id (must match samples.txt),

OS_time (positive float; survival time),

OS_event (0 = censored, 1 = event).

Optional: --outdir (default: outputs/level2_simple), --k (int, default: choose by silhouette over 2..8), --pca (int or 0; default 64), --cv (folds, default 5), --seed (default 42).

What to build (one file): scripts/level2_simple.py
Use only standard libs + numpy, pandas, scikit-learn, matplotlib, torch, and lifelines.

### Steps to implement

1. Load data

- Load embeddings tensor (torch.load), convert to numpy.
- Load samples.txt to get patient_id order.
- Load patient_meta.tsv, inner-join by patient_id, and align rows to the embeddings order. Assert no mismatch/missing.

2. Preprocess

- If --pca > 0: StandardScaler → PCA(n_components=--pca) to reduce embeddings before clustering.
- Always keep the original patient order.

3. Choose K (if not provided)

- For K in [2..8]: run KMeans (n_init=10, random_state=--seed) on the (possibly PCA-reduced) embeddings.
- Compute silhouette score; pick K with the highest score (tie-break by smaller K).

4. Final clustering

Run KMeans with the chosen K to get cluster_id per patient (0..K-1).

5. Survival verification with C-index (no cross-validation)

- Fit a single Cox model on the full cohort using lifelines.CoxPHFitter:
  - Build design matrix X from one-hot encoded cluster IDs (drop one baseline column to avoid collinearity).
  - Fit with cph.fit(df, duration_col="OS_time", event_col="OS_event").

Compute in-sample C-index:

Get risk scores with cph.predict_partial_hazard(df).

Compute Harrell’s C-index on the full data using lifelines.utils.concordance_index(OS_time, -risk, OS_event).

Report this single C-index value (note: it is optimistic because it’s in-sample).

Optional: bootstrap confidence interval (still no CV):

If --bootstrap N > 0, repeat N times:

Sample patients with replacement to form a bootstrap set.

Refit Cox on the bootstrap set; evaluate C-index on the original full cohort (or on the out-of-bag patients for slight optimism correction).

Report mean C-index and 95% CI from the bootstrap distribution.

Log-rank test (descriptive):

Run a log-rank test across clusters on the full data and print the p-value.

Outputs:

summary.json → { "c_index": ..., "c_index_boot_mean": ..., "c_index_boot_ci": [lo, hi], "logrank_p": ... }

Keep cluster_assignments.csv and plots as before.

6. Outputs

- Create --outdir if missing.
- Save cluster_assignments.csv with columns: patient_id, cluster_id.
- Save summary.json containing: k, silhouette, cv_c_index_mean, cv_c_index_std, logrank_p.
- Plot Kaplan–Meier curves per cluster (matplotlib) and save km_by_cluster.png.
- Plot a 2-D scatter for visualization:
  - If --pca > 1, plot the first two PCA components colored by cluster → pca2_scatter.png.

7. CLI

- Provide argparse for all flags above.
- Print a short run summary at the end with paths to saved artifacts.

8. Quality & robustness

- Set seeds (numpy, torch, sklearn).
- Clear error messages for missing columns/files or patient ID mismatches.
- Type hints and docstrings for main functions.
- Exit non-zero on failure.

### Example usage

```bash
python scripts/level2_simple.py \
 --embeddings outputs/tensors/patient_embeddings.pt \
 --patient-ids data/level1/samples.txt \
 --meta data/level2/patient_meta.tsv \
 --pca 64 --cv 5 --outdir outputs/level2_simple
```

### Success criteria

- Produces cluster_assignments.csv, summary.json, km_by_cluster.png, and (if PCA) pca2_scatter.png.
- Prints: chosen K, silhouette score, mean±std C-index (CV), and log-rank p-value.
- Runs in <1 minute on a small cohort (≤1k patients) on CPU.

### Nice-to-have (if trivial)

- Add --repeat to rerun KMeans multiple times and keep the best silhouette.
- Save fold-wise C-index values into cv_c_index.csv.
