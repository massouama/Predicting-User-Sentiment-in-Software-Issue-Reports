"""Generate (or regenerate) the cached synthetic dataset.

Standalone entry point so the corpus can be (re)built without running the full
experiment suite::

    python scripts/generate_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a script from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.data_generation import generate_dataset

if __name__ == "__main__":
    config.ensure_directories()
    df = generate_dataset()
    df.to_csv(config.DATASET_PATH, index=False)

    print(f"Wrote {len(df)} reports to {config.DATASET_PATH}")
    print("\nClass distribution:")
    print(df["label"].value_counts().to_string())
    print("\nExamples:")
    for _, row in df.head(8).iterrows():
        print(f"  [{row['label']:>8}] {row['text']}")
