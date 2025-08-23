from __future__ import annotations
import os, random, yaml, torch
from torch_geometric.loader import DataLoader
from ..utils.seed import set_seed
from ..utils.logging import CSVLogger
from ..dataio.dataset import PatientGraphDataset
from ..dataio.collate import hetero_collate
from ..models.hetero_encoder import HeteroGNN
from ..models.modality_pool import ModalityPooling
from ..models.modality_attention import ModalityAttention
from ..losses.recon_feature import FeatureDecoders, recon_feature_loss
from ..losses.recon_edge import edge_recon_losses
from ..losses.consistency import consistency_loss
from ..losses.entropy_reg import attention_entropy
from ..losses.weighting import normalize_lambda


def device_from_cfg(device_str: str):
    return (
        "cuda"
        if device_str == "auto" and torch.cuda.is_available()
        else device_str if device_str != "auto" else "cpu"
    )


class Level1Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        set_seed(cfg.get("seed", 42))
        self.device = torch.device(device_from_cfg(cfg.get("device", "auto")))
        dataset = PatientGraphDataset(cfg["data"])
        N = len(dataset)
        idx = list(range(N))
        if cfg["train"].get("shuffle_patients", True):
            random.shuffle(idx)
        n_val = max(1, int(N * cfg["train"].get("val_split", 0.2)))
        self.idx_train = idx[n_val:]
        self.idx_val = idx[:n_val]
        self.train_loader = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=True,
            collate_fn=hetero_collate,
        )
        self.val_loader = DataLoader(
            dataset,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            collate_fn=hetero_collate,
        )
        sample = dataset[0]
        metadata = sample.metadata()
        self.encoder = HeteroGNN(
            metadata,
            hidden_dim=cfg["model"]["hidden_dim"],
            out_dim=cfg["model"]["out_dim"],
            num_layers=cfg["model"]["num_layers"],
            conv=cfg["model"]["conv"],
            heads=cfg["model"]["heads"],
            dropout=cfg["model"]["dropout"],
            layernorm=cfg["model"]["layernorm"],
        ).to(self.device)
        node_out_dims = {nt: cfg["model"]["out_dim"] for nt in sample.node_types}
        self.pooler = ModalityPooling(node_out_dims, cfg).to(self.device)
        self.attn = ModalityAttention(
            dim=cfg["modalities"]["dim"], hidden=cfg["pooling"].get("attn_hidden", 64)
        ).to(self.device)
        orig_dims = {
            "gene": sample["gene"].x.size(-1),
            "cpg": sample["cpg"].x.size(-1),
            "mirna": sample["mirna"].x.size(-1),
        }
        self.decoders = FeatureDecoders(node_out_dims, orig_dims).to(self.device)
        self.optim = torch.optim.Adam(
            [
                {"params": self.encoder.parameters()},
                {"params": self.pooler.parameters()},
                {"params": self.attn.parameters()},
                {"params": self.decoders.parameters()},
            ],
            lr=cfg["train"]["lr"],
            weight_decay=cfg["train"]["weight_decay"],
        )
        self.lmbd = normalize_lambda(cfg["loss"]["lambda"])
        self.eta = cfg["loss"].get("recon_eta", 1.0)
        self.alpha = cfg["loss"].get("alpha_consistency", 1.0)
        self.beta = cfg["loss"].get("beta_entropy", 0.05)
        out_dir = cfg["export"]["out_dir"]
        os.makedirs(out_dir + "/logs", exist_ok=True)
        self.log = CSVLogger(
            out_dir + "/logs/train_log.csv",
            ["epoch", "split", "L_total", "L_recon", "L_cons", "L_ent"],
        )

    def _epoch(self, loader, train: bool):
        self.encoder.train(train)
        self.pooler.train(train)
        self.attn.train(train)
        self.decoders.train(train)
        sums = {"L_total": 0.0, "L_recon": 0.0, "L_cons": 0.0, "L_ent": 0.0}
        count = 0
        for data in loader:
            data = data.to(self.device)
            if train:
                self.optim.zero_grad()
            edge_index_dict = {}
            for rel in data.edge_types:
                if hasattr(data[rel], "edge_index"):
                    ei = data[rel].edge_index
                    # keep only relations with at least one edge
                    if ei is not None and ei.numel() > 0:
                        edge_index_dict[rel] = ei

            node_embeds = self.encoder(data.x_dict, edge_index_dict, None)
            feat_out = self.decoders(node_embeds, data)
            feat_losses = recon_feature_loss(feat_out, data.x_dict, data.batch_dict)
            edge_losses = edge_recon_losses(node_embeds, data, use_mlp=False)
            rel_to_mod = {}
            for rel in data.edge_types:
                if rel == ("cpg", "maps_to", "gene"):
                    rel_to_mod[str(rel)] = "DNAmeth"
                elif rel == ("mirna", "targets", "gene"):
                    rel_to_mod[str(rel)] = "miRNA"
                else:
                    rel_to_mod[str(rel)] = "mRNA"
            L_recon = 0.0
            per_mod = {
                "mRNA": feat_losses["mRNA"],
                "CNV": feat_losses["CNV"],
                "DNAmeth": feat_losses["DNAmeth"],
                "miRNA": feat_losses["miRNA"],
            }
            for m, loss_m in per_mod.items():
                tied = 0.0
                for rel_str, l in edge_losses.items():
                    if rel_str == "total":
                        continue
                    if rel_to_mod.get(rel_str) == m:
                        tied = tied + l
                L_recon = L_recon + self.lmbd[m] * (loss_m + self.eta * tied)
            modality_vecs = self.pooler(node_embeds, data.batch_dict)
            h, alpha, names = self.attn(modality_vecs)
            L_cons = consistency_loss(h, modality_vecs)
            L_ent = attention_entropy(alpha)
            L_total = L_recon + self.alpha * L_cons + self.beta * L_ent
            if train:
                L_total.backward()
                self.optim.step()
            for k, v in [
                ("L_total", L_total),
                ("L_recon", L_recon),
                ("L_cons", L_cons),
                ("L_ent", L_ent),
            ]:
                sums[k] += float(v.item())
            count += 1
        for k in sums:
            sums[k] /= max(count, 1)
        return sums

    def fit(self):
        results = {}
        # Stage A
        for epoch in range(1, self.cfg["train"]["max_epochs_stageA"] + 1):
            tr = self._epoch(self.train_loader, True)
            self.log.log({"epoch": epoch, "split": "train-A", **tr})
            va = self._epoch(self.val_loader, False)
            self.log.log({"epoch": epoch, "split": "val-A", **va})
        results["stageA_last"] = va
        # Stage B
        for epoch in range(1, self.cfg["train"]["max_epochs_stageB"] + 1):
            tr = self._epoch(self.train_loader, True)
            self.log.log({"epoch": epoch, "split": "train-B", **tr})
            va = self._epoch(self.val_loader, False)
            self.log.log({"epoch": epoch, "split": "val-B", **va})
        results["stageB_last"] = va
        self.export_embeddings()
        return results

    def export_embeddings(self):
        out_dir = self.cfg["export"]["out_dir"]
        os.makedirs(out_dir + "/tensors", exist_ok=True)
        self.encoder.eval()
        self.pooler.eval()
        self.attn.eval()
        all_patient_ids = []
        all_h = []
        all_alpha = []
        all_z = {"mRNA": [], "CNV": [], "DNAmeth": [], "miRNA": []}
        with torch.no_grad():
            for data in self.val_loader:  # use whole dataset loader for convenience
                data = data.to(self.device)
                node_embeds = self.encoder(data.x_dict, data.edge_index_dict, None)
                modality_vecs = self.pooler(node_embeds, data.batch_dict)
                h, alpha, names = self.attn(modality_vecs)
                all_h.append(h.cpu())
                all_alpha.append(alpha.cpu())
                for k in all_z.keys():
                    all_z[k].append(modality_vecs[k].cpu())
                # patient ids
                # Batch stores slice order; we stored patient_id per item, but PyG Batch loses list.
                # For export, we just enumerate rows.
                bsz = h.size(0)
                all_patient_ids += [
                    f"patient_{i}"
                    for i in range(len(all_patient_ids), len(all_patient_ids) + bsz)
                ]
        H = torch.cat(all_h, dim=0)
        torch.save(H, out_dir + "/tensors/patient_embeddings.pt")
        mod_embs = {k: torch.cat(v, dim=0) for k, v in all_z.items()}
        torch.save(mod_embs, out_dir + "/tensors/modality_embeddings.pt")
        # attention csv
        import pandas as pd

        A = torch.cat(all_alpha, dim=0).numpy()
        df = pd.DataFrame(A, columns=["mRNA", "CNV", "DNAmeth", "miRNA"])
        df.to_csv(out_dir + "/tensors/attention_weights.csv", index=False)
