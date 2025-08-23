#!/usr/bin/env python
"""
Test script to verify the Multi-Omics GNN implementation.
Run this after generating sample data to ensure everything works.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from pathlib import Path


def test_data_loading():
    """Test data loading functionality."""
    print("\n=== Testing Data Loading ===")

    from src.dataio.load_features import FeatureLoader
    from src.dataio.load_edges import EdgeLoader

    # Test feature loading
    try:
        feature_loader = FeatureLoader("data/features", "data/features/samples.txt")
        samples = feature_loader.samples
        print(f"✓ Loaded {len(samples)} patient samples")

        gene_features, gene_ids = feature_loader.load_features("genes.tsv", "gene")
        print(f"✓ Loaded gene features: {gene_features.shape}")

        cpg_features, cpg_ids = feature_loader.load_features("cpgs.tsv", "cpg")
        print(f"✓ Loaded CpG features: {cpg_features.shape}")

        mirna_features, mirna_ids = feature_loader.load_features("mirnas.tsv", "mirna")
        print(f"✓ Loaded miRNA features: {mirna_features.shape}")

    except Exception as e:
        print(f"✗ Feature loading failed: {e}")
        return False

    # Test edge loading
    try:
        edge_loader = EdgeLoader("data/edges")

        gene_cpg_edges = edge_loader.load_edges(
            "gene_cpg.csv", ("cpg", "maps_to", "gene")
        )
        print(f"✓ Loaded CpG-Gene edges: {len(gene_cpg_edges)} edges")

        gene_mirna_edges = edge_loader.load_edges(
            "gene_mirna.csv", ("mirna", "targets", "gene")
        )
        print(f"✓ Loaded miRNA-Gene edges: {len(gene_mirna_edges)} edges")

    except Exception as e:
        print(f"✗ Edge loading failed: {e}")
        return False

    return True


def test_graph_building():
    """Test graph construction."""
    print("\n=== Testing Graph Building ===")

    from src.dataio.load_features import FeatureLoader
    from src.dataio.load_edges import EdgeLoader
    from src.dataio.build_patient_graph import PatientGraphBuilder

    try:
        # Load data
        feature_loader = FeatureLoader("data/features", "data/features/samples.txt")
        edge_loader = EdgeLoader("data/edges")

        gene_features, gene_ids = feature_loader.load_features("genes.tsv", "gene")
        cpg_features, cpg_ids = feature_loader.load_features("cpgs.tsv", "cpg")
        mirna_features, mirna_ids = feature_loader.load_features("mirnas.tsv", "mirna")

        edges_data = {
            "gene_cpg": edge_loader.load_edges(
                "gene_cpg.csv", ("cpg", "maps_to", "gene")
            ),
            "gene_mirna": edge_loader.load_edges(
                "gene_mirna.csv", ("mirna", "targets", "gene")
            ),
        }

        # Build graph
        config = {"data": {"use_ppi_edges": False}}
        graph_builder = PatientGraphBuilder(
            gene_features,
            cpg_features,
            mirna_features,
            gene_ids,
            cpg_ids,
            mirna_ids,
            edges_data,
            config,
        )

        # Test for first patient
        patient_id = feature_loader.samples[0]
        graph = graph_builder.build_graph(patient_id)

        print(f"✓ Built graph for patient {patient_id}")
        print(f"  Gene nodes: {graph['gene'].x.shape}")
        print(f"  CpG nodes: {graph['cpg'].x.shape}")
        print(f"  miRNA nodes: {graph['mirna'].x.shape}")

        if ("cpg", "maps_to", "gene") in graph.edge_index_dict:
            edges = graph[("cpg", "maps_to", "gene")].edge_index
            print(f"  CpG->Gene edges: {edges.shape}")

        if ("mirna", "targets", "gene") in graph.edge_index_dict:
            edges = graph[("mirna", "targets", "gene")].edge_index
            print(f"  miRNA->Gene edges: {edges.shape}")

    except Exception as e:
        print(f"✗ Graph building failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_dataset():
    """Test dataset and dataloader."""
    print("\n=== Testing Dataset ===")

    from src.dataio.dataset import MultiOmicsDataset, get_dataloader

    try:
        # Create config
        config = {
            "seed": 42,
            "data": {
                "features_dir": "data/features/",
                "edges_dir": "data/edges/",
                "samples_file": "data/features/samples.txt",
                "genes_file": "genes.tsv",
                "cpgs_file": "cpgs.tsv",
                "mirnas_file": "mirnas.tsv",
                "gene_cpg_file": "gene_cpg.csv",
                "gene_mirna_file": "gene_mirna.csv",
                "gene_gene_file": "gene_gene.csv",
                "use_ppi_edges": False,
                "default_edge_weight": 1.0,
            },
            "training": {"val_split": 0.2, "batch_size": 4, "neg_sample_ratio": 5},
        }

        # Create dataset
        train_dataset = MultiOmicsDataset(config, split="train")
        val_dataset = MultiOmicsDataset(config, split="val")

        print(f"✓ Created datasets")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        # Test dataloader
        train_loader = get_dataloader(train_dataset, batch_size=4, shuffle=True)

        # Get one batch
        batch = next(iter(train_loader))
        print(f"✓ Created dataloader")
        print(f"  Batch gene features: {batch['gene'].x.shape}")
        print(f"  Batch patient IDs: {len(batch.patient_ids)}")

    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_model():
    """Test model initialization and forward pass."""
    print("\n=== Testing Model ===")

    from src.dataio.dataset import MultiOmicsDataset, get_dataloader
    from src.models.multiomics_gnn import MultiOmicsGNN

    try:
        # Create config
        config = {
            "seed": 42,
            "data": {
                "features_dir": "data/features/",
                "edges_dir": "data/edges/",
                "samples_file": "data/features/samples.txt",
                "genes_file": "genes.tsv",
                "cpgs_file": "cpgs.tsv",
                "mirnas_file": "mirnas.tsv",
                "gene_cpg_file": "gene_cpg.csv",
                "gene_mirna_file": "gene_mirna.csv",
                "gene_gene_file": "gene_gene.csv",
                "use_ppi_edges": False,
                "default_edge_weight": 1.0,
            },
            "model": {
                "hidden_dim": 64,
                "num_layers": 2,
                "num_heads": 2,
                "dropout": 0.1,
                "layer_norm": True,
                "concat_heads": False,
                "pooling_type": "mean",
                "attention_hidden": 32,
                "attention_dropout": 0.1,
            },
            "losses": {
                "feature_recon": 1.0,
                "edge_recon": 1.0,
                "consistency": 0.5,
                "entropy_reg": 0.1,
                "modality_weights": None,
            },
            "training": {"val_split": 0.2, "batch_size": 4, "neg_sample_ratio": 5},
        }

        # Create dataset
        dataset = MultiOmicsDataset(config, split="train")
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)

        # Get feature dimensions
        num_features_dict = dataset.get_num_features()

        # Create model
        model = MultiOmicsGNN(config, num_features_dict)
        print(f"✓ Created model")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Test forward pass
        device = torch.device("cpu")
        model = model.to(device)
        model.eval()

        batch = next(iter(dataloader))
        batch = batch.to(device)

        with torch.no_grad():
            results = model(batch, return_attention=True)

        print(f"✓ Forward pass successful")
        print(f"  Patient embedding shape: {results['patient_embedding'].shape}")
        print(f"  Modality vectors: {list(results['modality_vectors'].keys())}")

        # Test loss computation
        loss, loss_dict = model.compute_loss(results, batch, stage="pretrain")
        print(f"✓ Loss computation successful")
        print(f"  Total loss: {loss.item():.4f}")

    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_training_step():
    """Test a single training step."""
    print("\n=== Testing Training Step ===")

    from src.dataio.dataset import MultiOmicsDataset, get_dataloader
    from src.models.multiomics_gnn import MultiOmicsGNN
    import torch.optim as optim

    try:
        # Minimal config
        config = {
            "seed": 42,
            "data": {
                "features_dir": "data/features/",
                "edges_dir": "data/edges/",
                "samples_file": "data/features/samples.txt",
                "genes_file": "genes.tsv",
                "cpgs_file": "cpgs.tsv",
                "mirnas_file": "mirnas.tsv",
                "gene_cpg_file": "gene_cpg.csv",
                "gene_mirna_file": "gene_mirna.csv",
                "gene_gene_file": "gene_gene.csv",
                "use_ppi_edges": False,
                "default_edge_weight": 1.0,
            },
            "model": {
                "hidden_dim": 32,  # Small for testing
                "num_layers": 1,
                "num_heads": 2,
                "dropout": 0.1,
                "layer_norm": True,
                "concat_heads": False,
                "pooling_type": "mean",
                "attention_hidden": 16,
                "attention_dropout": 0.1,
            },
            "losses": {
                "feature_recon": 1.0,
                "edge_recon": 1.0,
                "consistency": 0.5,
                "entropy_reg": 0.1,
                "modality_weights": None,
            },
            "training": {
                "val_split": 0.2,
                "batch_size": 2,
                "neg_sample_ratio": 2,
                "learning_rate": 0.001,
            },
        }

        # Setup
        dataset = MultiOmicsDataset(config, split="train")
        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)
        num_features_dict = dataset.get_num_features()

        device = torch.device("cpu")
        model = MultiOmicsGNN(config, num_features_dict).to(device)
        optimizer = optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Training step
        model.train()
        batch = next(iter(dataloader)).to(device)

        optimizer.zero_grad()
        results = model(batch, return_attention=True)
        loss, loss_dict = model.compute_loss(results, batch, stage="full")
        loss.backward()
        optimizer.step()

        print(f"✓ Training step successful")
        print(f"  Loss: {loss.item():.4f}")
        print(
            f"  Loss components: {', '.join([f'{k}={v:.4f}' for k, v in loss_dict.items() if k != 'total'])}"
        )

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Multi-Omics GNN Implementation Test")
    print("=" * 50)

    # Check if sample data exists
    if not Path("data/features/samples.txt").exists():
        print("\n⚠ Sample data not found. Generating sample data first...")
        from scripts.generate_sample_data import generate_sample_data

        generate_sample_data(
            n_patients=50,  # Small dataset for testing
            n_genes=100,
            n_cpgs=80,
            n_mirnas=60,
            n_edges_cpg_gene=200,
            n_edges_mirna_gene=150,
            n_edges_gene_gene=50,
        )

    # Run tests
    tests = [
        ("Data Loading", test_data_loading),
        ("Graph Building", test_graph_building),
        ("Dataset", test_dataset),
        ("Model", test_model),
        ("Training Step", test_training_step),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n🎉 All tests passed! The implementation is ready to use.")
        print("\nNext steps:")
        print("1. Generate or prepare your real data in the required format")
        print("2. Adjust configuration in config/train.yaml")
        print("3. Run training: python scripts/train_level1.py")
    else:
        print("\n⚠ Some tests failed. Please check the error messages above.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
