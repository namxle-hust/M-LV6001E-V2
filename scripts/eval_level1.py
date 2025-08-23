# scripts/eval_level1.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

# Import necessary modules
from src.dataio.dataset import MultiOmicsDataset, get_dataloader
from src.models.multiomics_gnn import MultiOmicsGNN
from src.train.trainer import set_seed


def evaluate_reconstruction(model, data_loader, device):
    """Evaluate feature reconstruction performance."""
    model.eval()

    mse_per_type = {"gene": [], "cpg": [], "mirna": []}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating reconstruction"):
            batch = batch.to(device)

            # Forward pass
            results = model(batch)

            # Calculate MSE for each node type
            for node_type in mse_per_type.keys():
                pred = results["reconstructed_features"][node_type]
                target = batch[node_type].x

                mse = torch.nn.functional.mse_loss(pred, target, reduction="none")
                mse_per_type[node_type].append(mse.mean(dim=1).cpu().numpy())

    # Aggregate results
    results = {}
    for node_type, mse_list in mse_per_type.items():
        mse_array = np.concatenate(mse_list)
        results[f"MSE_{node_type}"] = {
            "mean": np.mean(mse_array),
            "std": np.std(mse_array),
            "median": np.median(mse_array),
        }

    return results


def evaluate_edge_prediction(model, data_loader, device):
    """Evaluate edge reconstruction performance."""
    model.eval()

    edge_metrics = {}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating edge prediction"):
            batch = batch.to(device)

            # Forward pass
            results = model(batch)

            # Evaluate each edge type
            for edge_type, edge_index in batch.edge_index_dict.items():
                if edge_index.shape[1] == 0:
                    continue

                src_type, _, dst_type = edge_type
                edge_type_str = f"{src_type}_{dst_type}"

                if edge_type_str not in edge_metrics:
                    edge_metrics[edge_type_str] = {"y_true": [], "y_pred": []}

                # Get embeddings
                src_embeddings = results["node_embeddings"][src_type][edge_index[0]]
                dst_embeddings = results["node_embeddings"][dst_type][edge_index[1]]

                # Positive predictions
                pos_pred = model.edge_decoder(
                    src_embeddings, dst_embeddings, edge_type_str
                )
                pos_pred = torch.sigmoid(pos_pred).cpu().numpy()

                # Sample negative edges
                neg_edge_index = model.edge_recon_loss.sample_negative_edges(
                    edge_index,
                    batch[src_type].num_nodes,
                    batch[dst_type].num_nodes,
                    device,
                )

                # Negative predictions
                neg_src_embeddings = results["node_embeddings"][src_type][
                    neg_edge_index[0]
                ]
                neg_dst_embeddings = results["node_embeddings"][dst_type][
                    neg_edge_index[1]
                ]
                neg_pred = model.edge_decoder(
                    neg_src_embeddings, neg_dst_embeddings, edge_type_str
                )
                neg_pred = torch.sigmoid(neg_pred).cpu().numpy()

                # Store predictions
                edge_metrics[edge_type_str]["y_true"].extend([1] * len(pos_pred))
                edge_metrics[edge_type_str]["y_true"].extend([0] * len(neg_pred))
                edge_metrics[edge_type_str]["y_pred"].extend(pos_pred.tolist())
                edge_metrics[edge_type_str]["y_pred"].extend(neg_pred.tolist())

    # Calculate metrics
    results = {}
    for edge_type, metrics in edge_metrics.items():
        y_true = np.array(metrics["y_true"])
        y_pred = np.array(metrics["y_pred"])

        results[edge_type] = {
            "AUROC": roc_auc_score(y_true, y_pred),
            "AUPRC": average_precision_score(y_true, y_pred),
        }

    return results


def analyze_attention_weights(attention_weights_path):
    """Analyze attention weight statistics."""
    df = pd.read_csv(attention_weights_path)

    # Calculate statistics
    modalities = ["mRNA", "CNV", "DNAmeth", "miRNA"]
    stats = {}

    for modality in modalities:
        weights = df[modality].values
        stats[modality] = {
            "mean": np.mean(weights),
            "std": np.std(weights),
            "min": np.min(weights),
            "max": np.max(weights),
            "median": np.median(weights),
        }

    # Calculate variance across patients
    patient_variance = df[modalities].var(axis=1).mean()

    return stats, patient_variance


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multi-Omics GNN Level-1")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Data split to evaluate on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]

    # Override batch size
    config["training"]["batch_size"] = args.batch_size

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set seed
    set_seed(config["seed"])

    # Create dataset
    print(f"\nLoading {args.data_split} dataset...")
    dataset = MultiOmicsDataset(config, split=args.data_split)
    data_loader = get_dataloader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print("Initializing model...")
    num_features_dict = dataset.get_num_features()
    model = MultiOmicsGNN(config, num_features_dict)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Validation loss at checkpoint: {checkpoint['val_losses']['total']:.4f}")

    # Evaluate reconstruction
    print("\n=== Feature Reconstruction Evaluation ===")
    recon_results = evaluate_reconstruction(model, data_loader, device)

    for node_type, metrics in recon_results.items():
        print(f"\n{node_type}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.6f}")

    # Evaluate edge prediction
    print("\n=== Edge Prediction Evaluation ===")
    edge_results = evaluate_edge_prediction(model, data_loader, device)

    for edge_type, metrics in edge_results.items():
        print(f"\n{edge_type}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")

    # Analyze attention weights if available
    attention_path = Path(config["outputs"]["attention_weights_path"])
    if attention_path.exists():
        print("\n=== Attention Weight Analysis ===")
        attention_stats, patient_var = analyze_attention_weights(attention_path)

        for modality, stats in attention_stats.items():
            print(f"\n{modality}:")
            for stat_name, value in stats.items():
                print(f"  {stat_name}: {value:.4f}")

        print(f"\nAverage variance across patients: {patient_var:.4f}")

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
