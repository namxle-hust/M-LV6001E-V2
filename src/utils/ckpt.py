from __future__ import annotations
import os, torch


def save_checkpoint(model, path: str, extra: dict | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"state_dict": model.state_dict()}
    if extra:
        obj.update(extra)
    torch.save(obj, path)


def load_checkpoint(model, path: str, map_location=None):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return ckpt
