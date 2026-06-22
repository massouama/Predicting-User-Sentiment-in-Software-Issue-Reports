"""Exploratory data analysis for the issue-report corpus.

Prints summary statistics (class balance, document lengths, per-class examples)
and writes the EDA figures used in the report::

    python scripts/explore_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.balancing import class_distribution
from src.data_generation import load_or_create_dataset
from src.experiment import run_eda
from src.preprocessing import TextPreprocessor

if __name__ == "__main__":
    config.ensure_directories()
    df = load_or_create_dataset()
    df["clean"] = TextPreprocessor().transform(df["text"].tolist())

    print(f"Corpus size: {len(df)} reports\n")

    dist = class_distribution(df["label"])
    print("Class distribution:")
    for cls, n in dist.items():
        print(f"  {cls:<8} {n:>5}  ({n / len(df):.1%})")

    tok = df["clean"].str.split().map(len)
    print(f"\nCleaned token count -- min={tok.min()} max={tok.max()} mean={tok.mean():.1f}")

    print("\nOne example per class:")
    for cls in config.CLASS_NAMES:
        example = df.loc[df["label"] == cls, "text"].iloc[0]
        print(f"  [{cls:>8}] {example}")

    run_eda(df)
    print(f"\nFigures written to {config.FIGURES_DIR}")
