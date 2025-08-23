from __future__ import annotations
from torch.utils.data import Dataset
from .load_features import load_feature_tables
from .load_edges import load_edge_tables
from .build_patient_graph import build_patient_hetero_graph


class PatientGraphDataset(Dataset):
    def __init__(self, cfg_data: dict):
        self.tables = load_feature_tables(cfg_data)
        self.edge_dfs = load_edge_tables(cfg_data)
        self.patients = self.tables["patients"]

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx: int):
        data = build_patient_hetero_graph(idx, self.tables, self.edge_dfs)
        data.patient_id = self.patients[idx]
        return data
