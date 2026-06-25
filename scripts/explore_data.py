"""Analyse exploratoire du corpus d'avis.

Affiche des statistiques descriptives (équilibre des classes, longueurs des
documents, exemples par classe) et écrit les figures d'EDA du rapport ::

    python scripts/explore_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.balancing import class_distribution
from src.real_data import load_or_create_dataset
from src.experiment import run_eda
from src.preprocessing import TextPreprocessor

if __name__ == "__main__":
    config.ensure_directories()
    df = load_or_create_dataset()
    df["clean"] = TextPreprocessor().transform(df["text"].tolist())

    print(f"Taille du corpus : {len(df)} avis\n")

    dist = class_distribution(df["label"])
    print("Distribution des classes :")
    for cls, n in dist.items():
        print(f"  {cls:<8} {n:>5}  ({n / len(df):.1%})")

    tok = df["clean"].str.split().map(len)
    print(f"\nNombre de tokens nettoyés -- min={tok.min()} max={tok.max()} moyenne={tok.mean():.1f}")

    print("\nUn exemple par classe :")
    for cls in config.CLASS_NAMES:
        example = df.loc[df["label"] == cls, "text"].iloc[0]
        print(f"  [{cls:>8}] {example}")

    run_eda(df)
    print(f"\nFigures écrites dans {config.FIGURES_DIR}")
