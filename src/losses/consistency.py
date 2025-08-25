"""
Consistency loss between fused and modality embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class ConsistencyLoss(nn.Module):
    """
    Encourage consistency between fused patient embedding and individual modality embeddings.
    """

    def __init__(self, distance_metric: str = "l2", normalize: bool = True):
        """
        Initialize consistency loss.

        Args:
            distance_metric: Distance metric ('l2', 'cosine', 'l1')
            normalize: Whether to normalize embeddings before computing distance
        """
        super().__init__()

        self.distance_metric = distance_metric
        self.normalize = normalize

    def compute_distance(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """
        Compute distance between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Distance value
        """
        if self.normalize:
            emb1 = F.normalize(emb1, dim=-1)
            emb2 = F.normalize(emb2, dim=-1)

        if self.distance_metric == "l2":
            distance = torch.norm(emb1 - emb2, p=2, dim=-1).mean()
        elif self.distance_metric == "l1":
            distance = torch.norm(emb1 - emb2, p=1, dim=-1).mean()
        elif self.distance_metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)
            distance = (1 - cos_sim).mean()
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        return distance

    def forward(
        self,
        fused_embedding: torch.Tensor,
        modality_embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute consistency loss.

        Args:
            fused_embedding: Fused patient embedding [batch_size, hidden_size]
            modality_embeddings: Individual modality embeddings

        Returns:
            - Total consistency loss
            - Individual distances by modality
        """
        distances = {}

        for modality, emb in modality_embeddings.items():
            distance = self.compute_distance(fused_embedding, emb)
            distances[modality] = distance

        # Average distance across modalities
        total_loss = sum(distances.values()) / len(distances)

        return total_loss, distances


class EntropyRegularization(nn.Module):
    """
    Entropy regularization to prevent attention collapse.
    """

    def __init__(self, target_entropy: Optional[float] = None, epsilon: float = 1e-8):
        """
        Initialize entropy regularization.

        Args:
            target_entropy: Target entropy value (None for maximum entropy)
            epsilon: Small value for numerical stability
        """
        super().__init__()

        self.target_entropy = target_entropy
        self.epsilon = epsilon

    def compute_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention weights.

        Args:
            attention_weights: Attention weights [batch_size, num_modalities]

        Returns:
            Entropy value
        """
        # Add epsilon for numerical stability
        weights = attention_weights + self.epsilon

        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(weights * torch.log(weights), dim=-1)

        return entropy.mean()

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss.

        Args:
            attention_weights: Attention weights [batch_size, num_modalities]

        Returns:
            Entropy loss (negative entropy to maximize it)
        """
        entropy = self.compute_entropy(attention_weights)

        if self.target_entropy is not None:
            # Penalize deviation from target entropy
            loss = torch.abs(entropy - self.target_entropy)
        else:
            # Maximize entropy (minimize negative entropy)
            loss = -entropy

        return loss


class AlignmentLoss(nn.Module):
    """
    Alignment loss to ensure modality embeddings are in the same space.
    """

    def __init__(self, use_adversarial: bool = False):
        """
        Initialize alignment loss.

        Args:
            use_adversarial: Whether to use adversarial alignment
        """
        super().__init__()

        self.use_adversarial = use_adversarial

        if use_adversarial:
            # Discriminator to distinguish between modalities
            self.discriminator = nn.Sequential(
                nn.Linear(256, 128),  # Assuming hidden_size=256
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 4),  # 4 modalities
            )

    def forward(self, modality_embeddings: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute alignment loss.

        Args:
            modality_embeddings: Modality embeddings

        Returns:
            Alignment loss
        """
        if not self.use_adversarial:
            # Simple alignment: minimize variance between modality embeddings
            embeddings_list = list(modality_embeddings.values())

            if len(embeddings_list) < 2:
                return torch.tensor(0.0, device=embeddings_list[0].device)

            # Stack embeddings
            stacked = torch.stack(
                embeddings_list, dim=0
            )  # [num_modalities, batch_size, hidden_size]

            # Compute mean across modalities
            mean_emb = stacked.mean(dim=0, keepdim=True)

            # Compute variance
            variance = ((stacked - mean_emb) ** 2).mean()

            return variance
        else:
            # Adversarial alignment
            loss = 0.0
            modality_labels = {"mRNA": 0, "CNV": 1, "DNAmeth": 2, "miRNA": 3}

            all_embeddings = []
            all_labels = []

            for modality, emb in modality_embeddings.items():
                all_embeddings.append(emb)
                labels = torch.full(
                    (emb.size(0),),
                    modality_labels[modality],
                    dtype=torch.long,
                    device=emb.device,
                )
                all_labels.append(labels)

            all_embeddings = torch.cat(all_embeddings, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Discriminator loss (we want it to fail)
            logits = self.discriminator(all_embeddings)

            # Use uniform distribution as target (maximum confusion)
            uniform_target = torch.ones_like(logits) / logits.size(1)
            loss = F.kl_div(
                F.log_softmax(logits, dim=-1), uniform_target, reduction="batchmean"
            )

            return -loss  # Negative because we want to maximize confusion
