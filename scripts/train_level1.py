from __future__ import annotations
import argparse, yaml
from src.train.trainer import Level1Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/train.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    trainer = Level1Trainer(cfg)
    results = trainer.fit()
    print("Finished. Last:", results)


if __name__ == "__main__":
    main()
