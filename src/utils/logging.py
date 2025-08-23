from __future__ import annotations
import os, csv


class CSVLogger:
    def __init__(self, path: str, fieldnames: list[str]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        if write_header:
            self.w.writeheader()

    def log(self, row: dict):
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()
