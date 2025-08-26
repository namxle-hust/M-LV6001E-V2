"""
Evaluation script for trained Level-1 model.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataio.load_features import FeatureLoader
from src.dataio.load_edges import EdgeLoader
from src.dataio.build_patient_graph import PatientGraphBuilder
from src.dataio.dataset import MultiModalDataModule
from src.models.multimodal_gnn import MultiModalGNNWithDecoders
from src.losses.recon_edge import LinkPredictionMetrics


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model = MultiModalGNNWithDecoders(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, config, checkpoint


def evaluate_reconstruction(
    model: torch.nn.Module, dataloader, device: torch.device
) -> dict:
    """
    Evaluate reconstruction quality.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device

    Returns:
        Dictionary of reconstruction metrics
    """
    model.eval()

    all_metrics = {
        "mrna_mse": [],
        "mrna_mae": [],
        "cnv_mse": [],
        "cnv_mae": [],
        "cpg_mse": [],
        "cpg_mae": [],
        "mirna_mse": [],
        "mirna_mae": [],
    }

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating reconstruction"):
            data = data.to(device)

            # Forward pass
            output = model(data, compute_loss=False)

            # Decode features
            reconstructed = model.feature_decoders(output["node_embeddings"])

            # Compare with original
            # mrna
            if "gene_mrna" in reconstructed:
                pred = reconstructed["gene_mrna"].squeeze().cpu().numpy()
                true = data.gene_mrna_batched.cpu().numpy()
                all_metrics["mrna_mse"].append(mean_squared_error(true, pred))
                all_metrics["mrna_mae"].append(mean_absolute_error(true, pred))

            # cnv
            if "gene_cnv" in reconstructed:
                pred = reconstructed["gene_cnv"].squeeze().cpu().numpy()
                true = data.gene_cnv_batched.cpu().numpy()
                all_metrics["cnv_mse"].append(mean_squared_error(true, pred))
                all_metrics["cnv_mae"].append(mean_absolute_error(true, pred))

            # cpg
            if "cpg" in reconstructed:
                pred = reconstructed["cpg"].squeeze().cpu().numpy()
                true = data["cpg"].x.squeeze().cpu().numpy()
                all_metrics["cpg_mse"].append(mean_squared_error(true, pred))
                all_metrics["cpg_mae"].append(mean_absolute_error(true, pred))

            # mirna
            if "mirna" in reconstructed:
                pred = reconstructed["mirna"].squeeze().cpu().numpy()
                true = data["mirna"].x.squeeze().cpu().numpy()
                all_metrics["mirna_mse"].append(mean_squared_error(true, pred))
                all_metrics["mirna_mae"].append(mean_absolute_error(true, pred))

    # Average metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

    return avg_metrics


def evaluate_link_prediction(
    model: torch.nn.Module, dataloader, device: torch.device
) -> dict:
    """
    Evaluate link prediction quality.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device

    Returns:
        Dictionary of link prediction metrics
    """
    model.eval()

    link_metrics = {}
    edge_types = [("cpg", "maps_to", "gene"), ("mirna", "targets", "gene")]

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating link prediction"):
            data = data.to(device)

            # Forward pass
            output = model(data, compute_loss=False)

            # Evaluate each edge type
            for edge_type in edge_types:
                if not hasattr(data[edge_type], "edge_index"):
                    continue

                src_type, rel_type, dst_type = edge_type
                edge_key = f"{src_type}_{rel_type}_{dst_type}"

                # Get embeddings
                z_src = output["node_embeddings"][src_type]
                z_dst = output["node_embeddings"][dst_type]

                # Get decoder
                decoder = model.edge_decoder.get_decoder(edge_type)

                # Positive edges
                edge_index = data[edge_type].edge_index
                pos_src = z_src[edge_index[0]]
                pos_dst = z_dst[edge_index[1]]
                pos_pred = torch.sigmoid(decoder(pos_src, pos_dst))

                # Sample negative edges
                neg_edge_index = model.edge_decoder.sample_negative_edges(
                    edge_index, z_src.shape[0], z_dst.shape[0], edge_index.shape[1]
                )

                neg_src = z_src[neg_edge_index[0]]
                neg_dst = z_dst[neg_edge_index[1]]
                neg_pred = torch.sigmoid(decoder(neg_src, neg_dst))

                # Compute metrics
                metrics = LinkPredictionMetrics.compute_auc_ap(pos_pred, neg_pred)

                if edge_key not in link_metrics:
                    link_metrics[edge_key] = {"auroc": [], "auprc": []}

                link_metrics[edge_key]["auroc"].append(metrics["auroc"])
                link_metrics[edge_key]["auprc"].append(metrics["auprc"])

    # Average metrics
    avg_link_metrics = {}
    for edge_key, metrics in link_metrics.items():
        avg_link_metrics[f"{edge_key}_auroc"] = np.mean(metrics["auroc"])
        avg_link_metrics[f"{edge_key}_auprc"] = np.mean(metrics["auprc"])

    return avg_link_metrics


def analyze_attention_weights(attention_weights_path: str, output_dir: str):
    """
    Analyze and visualize attention weights.

    Args:
        attention_weights_path: Path to attention weights CSV
        output_dir: Directory to save visualizations
    """
    # Load attention weights
    df = pd.read_csv(attention_weights_path, index_col=0)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Summary statistics
    summary = df.describe()
    summary.to_csv(os.path.join(output_dir, "attention_summary.csv"))

    print("\nAttention Weight Summary:")
    print(summary)

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Distribution of attention weights
    df.boxplot(ax=axes[0, 0])
    axes[0, 0].set_title("Distribution of Attention Weights by Modality")
    axes[0, 0].set_ylabel("Attention Weight")

    # 2. Correlation between modality weights
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=axes[0, 1])
    axes[0, 1].set_title("Correlation between Modality Attention Weights")

    # 3. Average attention per modality
    avg_attention = df.mean()
    avg_attention.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title("Average Attention Weight per Modality")
    axes[1, 0].set_ylabel("Average Weight")
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45)

    # 4. Attention weight variance
    var_attention = df.var()
    var_attention.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title("Variance of Attention Weights per Modality")
    axes[1, 1].set_ylabel("Variance")
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "attention_analysis.png"), dpi=300)
    plt.show()

    # Identify patients with extreme attention patterns
    for modality in df.columns:
        top_patients = df.nlargest(5, modality)
        print(f"\nTop 5 patients with highest {modality} attention:")
        print(top_patients[[modality]])


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Level-1 model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="Path to data directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation/",
        help="Directory for evaluation outputs",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )

    args = parser.parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    model, config, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}, stage {checkpoint['stage']}")
    print(f"Validation metrics: {checkpoint['metrics']}")

    # Update config with data directory
    config["data"]["features_dir"] = os.path.join(args.data_dir, "features/")
    config["data"]["edges_dir"] = os.path.join(args.data_dir, "edges/")
    config["data"]["samples_file"] = os.path.join(args.data_dir, "features/samples.txt")

    # Load data
    print("\nLoading data...")
    feature_loader = FeatureLoader(config["data"]["features_dir"])
    features_dict = feature_loader.load_all_features(config)

    edge_loader = EdgeLoader(config["data"]["edges_dir"])
    edges_dict = edge_loader.load_all_edges(config, features_dict["node_ids"])

    # Build graphs
    print("Building patient graphs...")
    graph_builder = PatientGraphBuilder(features_dict, edges_dict)
    all_graphs = graph_builder.build_all_graphs()

    # Create data module
    data_module = MultiModalDataModule(all_graphs, config)
    test_loader = data_module.test_dataloader()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Evaluate reconstruction
    print("\n=== Evaluating Reconstruction Quality ===")
    recon_metrics = evaluate_reconstruction(model, test_loader, device)

    print("\nReconstruction Metrics:")
    for key, value in recon_metrics.items():
        if not key.endswith("_std"):
            print(f"  {key}: {value:.4f}")

    # Evaluate link prediction
    print("\n=== Evaluating Link Prediction ===")
    link_metrics = evaluate_link_prediction(model, test_loader, device)

    print("\nLink Prediction Metrics:")
    for key, value in link_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Analyze attention weights if available
    attention_path = os.path.join(
        config["logging"]["tensors_dir"], "attention_weights.csv"
    )
    if os.path.exists(attention_path):
        print("\n=== Analyzing Attention Weights ===")
        analyze_attention_weights(attention_path, args.output_dir)

    # Save all metrics
    all_metrics = {
        "reconstruction": recon_metrics,
        "link_prediction": link_metrics,
        "checkpoint_info": {
            "path": args.checkpoint,
            "epoch": checkpoint["epoch"],
            "stage": checkpoint["stage"],
            "validation_metrics": checkpoint["metrics"],
        },
    }

    metrics_file = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nEvaluation metrics saved to {metrics_file}")

    # Create summary report
    report = f"""
    ========================================
    Level-1 Model Evaluation Report
    ========================================
    
    Model Information:
    - Checkpoint: {args.checkpoint}
    - Epoch: {checkpoint['epoch']}
    - Stage: {checkpoint['stage']}
    
    Reconstruction Performance:
    - mrna MSE: {recon_metrics.get('mrna_mse', 'N/A'):.4f}
    - cnv MSE: {recon_metrics.get('cnv_mse', 'N/A'):.4f}
    - cpg MSE: {recon_metrics.get('cpg_mse', 'N/A'):.4f}
    - mirna MSE: {recon_metrics.get('mirna_mse', 'N/A'):.4f}
    
    Link Prediction Performance:
    - cpg→gene AUROC: {link_metrics.get('cpg_maps_to_gene_auroc', 'N/A'):.4f}
    - mirna→gene AUROC: {link_metrics.get('mirna_targets_gene_auroc', 'N/A'):.4f}
    
    ========================================
    """

    print(report)

    # Save report
    report_file = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Evaluation report saved to {report_file}")


if __name__ == "__main__":
    main()
