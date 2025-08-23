from __future__ import annotations


def normalize_lambda(lmbd: dict[str, float]) -> dict[str, float]:
    s = sum(lmbd.values())
    return {k: (v / s if s > 0 else 1.0 / len(lmbd)) for k, v in lmbd.items()}
