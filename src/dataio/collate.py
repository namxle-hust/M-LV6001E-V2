from __future__ import annotations
from torch_geometric.data import Batch


def hetero_collate(batch_list):
    return Batch.from_data_list(batch_list)
