I am planing design a hierarchical heterogeneous graph neural network for multi-omics cancer subtyping with the following architecture. Treated this as a unsupervised clustering task with no label, we don't know what patient should be cluster with any patient. Finaly evaluate the clustering result with C-index and p-value. Based on paper CtAE and M2GGCN and any paper related in this field, don't generate any codes, what do you think I should improve?

```
### Level 1: Patient-Specific Heterogeneous Graphs

- Node Types: 4 omics modalities per patient (mRNA, CNV, DNA methylation, miRNA)
- Intra-Patient Edges: Connect omics nodes through shared Ensembl gene IDs
  - mRNA ↔ CNV: Direct Ensembl ID mapping
  - DNA methylation → Genes: CpG site genomic coordinate mapping to nearest TSS
  - miRNA → Genes: Target prediction (TargetScan + miRTarBase validation)
- Node Features: All preprocessed features after the preprocessv3.py script, without selecting top features.

### Level 2: Inter-Patient Pathway Network

- Patient Nodes: Each patient becomes a node in the global network
- Edge Types:
  1. Shared Pathway Alteration: Patients connected if they have alterations in the same pathways
  2. Gene-Specific Alteration: Patients connected through specific genes with same alteration types (CNA amplification, deletion, mutation, overexpression)
- Edge Weights: Based on:

  - Frequency of shared pathway alterations
  - Similarity of alteration patterns
  - Clinical significance of shared genes

### Technical Implementation

- Architecture: Heterogeneous Graph Attention Network (HeteroGAT) for superior performance on cancer data
- Pathway Data: Integrate cBioPortal pathway information with KEGG/Reactome databases
- Alteration Mapping: Use cBioPortal alteration annotations (amplified, deleted, mutated, overexpressed)
- Message Passing: Multi-level attention mechanisms for both intra-patient omics integration and inter-patient pathway relationships

### Expected Outputs

- Patient-level cancer subtype predictions
- Pathway-level importance scores
- Gene-level biomarker identification
- Interpretable attention weights for clinical relevance

```
