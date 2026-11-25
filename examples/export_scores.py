"""Small CPU-authored torch.export input for the offline compiler CLI."""

import argparse
from pathlib import Path
import torch
from examples.graphs import ScoreSoftmax

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    a = p.parse_args()
    if a.archive.exists():
        p.error("archive exists; choose a new path")
    x = torch.zeros(12, 129, dtype=torch.bfloat16)
    b = torch.zeros(129, dtype=x.dtype)
    torch.export.save(torch.export.export(ScoreSoftmax(True), (x, b)), a.archive)
