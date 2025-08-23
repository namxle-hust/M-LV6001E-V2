#!/usr/bin/env python
"""
Generate sample multi-omics data for testing the implementation.
This creates synthetic data in the required format.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def generate_sample_data(
    n_patients: int = 100,
    n_genes: int = 500,
    n_cpgs: int = 300,
    n_mirnas: int = 200,
    n_edges_cpg_gene: int = 1000,
    n_edges_mirna_gene: int = 500,
    n_edges_gene_gene: int = 200,
    output_dir: str = "data",
    seed: int = 42,
):
    """Generate synthetic multi-omics data."""

    np.random.seed(seed)

    # Create output directories
    output_path = Path(output_dir)
    features_path = output_path / "features"
    edges_path = output_path / "edges"

    features_path.mkdir(parents=True, exist_ok=True)
    edges_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating sample data with {n_patients} patients...")

    # Generate patient IDs
    patient_ids = [f"Patient_{i:04d}" for i in range(1, n_patients + 1)]

    # Save patient IDs
    samples_file = features_path / "samples.txt"
    with open(samples_file, "w") as f:
        for pid in patient_ids:
            f.write(f"{pid}\n")
    print(f"Saved patient IDs to {samples_file}")

    # Generate gene IDs
    gene_ids = [f"GENE_{i:04d}" for i in range(1, n_genes + 1)]

    # Generate CpG IDs
    cpg_ids = [f"cg{i:08d}" for i in range(1, n_cpgs + 1)]

    # Generate miRNA IDs
    mirna_ids = [f"hsa-mir-{i:03d}" for i in range(1, n_mirnas + 1)]

    # Generate gene expression data (mRNA + CNV combined)
    print("Generating gene expression data...")
    gene_data = np.random.randn(n_genes, n_patients)

    # Add some structure - make some genes correlated
    n_modules = 10
    genes_per_module = n_genes // n_modules
    for i in range(n_modules):
        start_idx = i * genes_per_module
        end_idx = min((i + 1) * genes_per_module, n_genes)

        # Create correlated expression within modules
        module_factor = np.random.randn(1, n_patients)
        gene_data[start_idx:end_idx] += 0.5 * module_factor

    # Normalize
    gene_data = (gene_data - gene_data.mean(axis=1, keepdims=True)) / (
        gene_data.std(axis=1, keepdims=True) + 1e-8
    )

    # Create DataFrame
    gene_df = pd.DataFrame(gene_data, index=gene_ids, columns=patient_ids)
    gene_df.to_csv(features_path / "genes.tsv", sep="\t")
    print(f"Saved gene expression to {features_path / 'genes.tsv'}")

    # Generate DNA methylation data
    print("Generating DNA methylation data...")
    # Methylation values are typically between 0 and 1, but we'll use normalized values
    cpg_data = np.random.beta(2, 5, size=(n_cpgs, n_patients))

    # Add some missing values (methylation data often has missing values)
    mask = np.random.random((n_cpgs, n_patients)) < 0.05  # 5% missing
    cpg_data[mask] = np.nan

    # Normalize
    cpg_data = (cpg_data - np.nanmean(cpg_data, axis=1, keepdims=True)) / (
        np.nanstd(cpg_data, axis=1, keepdims=True) + 1e-8
    )

    cpg_df = pd.DataFrame(cpg_data, index=cpg_ids, columns=patient_ids)
    cpg_df.to_csv(features_path / "cpgs.tsv", sep="\t")
    print(f"Saved DNA methylation to {features_path / 'cpgs.tsv'}")

    # Generate miRNA expression data
    print("Generating miRNA expression data...")
    # miRNA expression is often sparse
    mirna_data = np.random.exponential(scale=0.5, size=(n_mirnas, n_patients))
    mirna_data[mirna_data < 0.3] = 0  # Make sparse

    # Log transform and normalize
    mirna_data = np.log1p(mirna_data)
    mirna_data = (mirna_data - mirna_data.mean(axis=1, keepdims=True)) / (
        mirna_data.std(axis=1, keepdims=True) + 1e-8
    )

    mirna_df = pd.DataFrame(mirna_data, index=mirna_ids, columns=patient_ids)
    mirna_df.to_csv(features_path / "mirnas.tsv", sep="\t")
    print(f"Saved miRNA expression to {features_path / 'mirnas.tsv'}")

    # Generate edge files
    print("\nGenerating edge relationships...")

    # CpG -> Gene edges (methylation sites mapped to genes)
    cpg_gene_edges = []
    for _ in range(n_edges_cpg_gene):
        cpg = np.random.choice(cpg_ids)
        gene = np.random.choice(gene_ids)
        weight = np.random.uniform(0.5, 1.0)
        cpg_gene_edges.append([cpg, gene, weight])

    cpg_gene_df = pd.DataFrame(cpg_gene_edges, columns=["cpg_id", "gene_id", "weight"])
    # Remove duplicates
    cpg_gene_df = cpg_gene_df.drop_duplicates(subset=["cpg_id", "gene_id"])
    cpg_gene_df.to_csv(edges_path / "gene_cpg.csv", index=False)
    print(f"Saved {len(cpg_gene_df)} CpG-Gene edges to {edges_path / 'gene_cpg.csv'}")

    # miRNA -> Gene edges (regulatory relationships)
    mirna_gene_edges = []
    for _ in range(n_edges_mirna_gene):
        mirna = np.random.choice(mirna_ids)
        gene = np.random.choice(gene_ids)
        weight = np.random.uniform(0.5, 1.0)
        mirna_gene_edges.append([mirna, gene, weight])

    mirna_gene_df = pd.DataFrame(
        mirna_gene_edges, columns=["mirna_id", "gene_id", "weight"]
    )
    mirna_gene_df = mirna_gene_df.drop_duplicates(subset=["mirna_id", "gene_id"])
    mirna_gene_df.to_csv(edges_path / "gene_mirna.csv", index=False)
    print(
        f"Saved {len(mirna_gene_df)} miRNA-Gene edges to {edges_path / 'gene_mirna.csv'}"
    )

    # Gene -> Gene edges (PPI network, optional)
    if n_edges_gene_gene > 0:
        gene_gene_edges = []
        for _ in range(n_edges_gene_gene):
            gene1 = np.random.choice(gene_ids)
            gene2 = np.random.choice(gene_ids)
            if gene1 != gene2:  # No self-loops
                weight = np.random.uniform(0.5, 1.0)
                gene_gene_edges.append([gene1, gene2, weight])

        gene_gene_df = pd.DataFrame(
            gene_gene_edges, columns=["gene1_id", "gene2_id", "weight"]
        )
        gene_gene_df = gene_gene_df.drop_duplicates(subset=["gene1_id", "gene2_id"])
        gene_gene_df.to_csv(edges_path / "gene_gene.csv", index=False)
        print(
            f"Saved {len(gene_gene_df)} Gene-Gene edges to {edges_path / 'gene_gene.csv'}"
        )

    print("\nSample data generation complete!")
    print(f"\nData summary:")
    print(f"  Patients: {n_patients}")
    print(f"  Genes: {n_genes}")
    print(f"  CpG sites: {n_cpgs}")
    print(f"  miRNAs: {n_mirnas}")
    print(f"  CpG-Gene edges: {len(cpg_gene_df)}")
    print(f"  miRNA-Gene edges: {len(mirna_gene_df)}")
    if n_edges_gene_gene > 0:
        print(f"  Gene-Gene edges: {len(gene_gene_df)}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Generate sample multi-omics data")
    parser.add_argument(
        "--n_patients", type=int, default=100, help="Number of patients"
    )
    parser.add_argument("--n_genes", type=int, default=500, help="Number of genes")
    parser.add_argument("--n_cpgs", type=int, default=300, help="Number of CpG sites")
    parser.add_argument("--n_mirnas", type=int, default=200, help="Number of miRNAs")
    parser.add_argument(
        "--n_edges_cpg_gene", type=int, default=1000, help="Number of CpG-Gene edges"
    )
    parser.add_argument(
        "--n_edges_mirna_gene", type=int, default=500, help="Number of miRNA-Gene edges"
    )
    parser.add_argument(
        "--n_edges_gene_gene",
        type=int,
        default=200,
        help="Number of Gene-Gene edges (PPI)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data", help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    generate_sample_data(
        n_patients=args.n_patients,
        n_genes=args.n_genes,
        n_cpgs=args.n_cpgs,
        n_mirnas=args.n_mirnas,
        n_edges_cpg_gene=args.n_edges_cpg_gene,
        n_edges_mirna_gene=args.n_edges_mirna_gene,
        n_edges_gene_gene=args.n_edges_gene_gene,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
