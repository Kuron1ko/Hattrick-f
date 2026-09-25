from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch


def archive_once(run_dir: Path) -> None:
    source = run_dir / "final_model.pt"
    if not source.exists():
        return
    try:
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError):
        return
    epoch = int(checkpoint["epoch"])
    target_dir = run_dir / "epoch_checkpoints"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"epoch_{epoch:03d}.pt"
    if not target.exists():
        shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive each completed training epoch")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    while not (run_dir / "complete.json").exists():
        archive_once(run_dir)
        time.sleep(1.0)
    archive_once(run_dir)


if __name__ == "__main__":
    main()
