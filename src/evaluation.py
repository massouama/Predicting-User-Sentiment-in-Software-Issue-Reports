"""Métriques et fonctions de visualisation.

Regroupe tout ce que le sujet demande sous « Évaluation » — accuracy, précision,
rappel, F1 et matrice de confusion — ainsi que les graphiques de comparaison du
rapport. La moyenne *macro* est utilisée partout : sur un problème déséquilibré,
elle pondère chaque classe également et reflète donc la performance sur les
classes minoritaires (que l'accuracy masque).

Un backend Matplotlib non interactif est choisi à l'import pour produire les
figures sur un serveur sans affichage.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # rendu sans affichage ; doit précéder l'import de pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Renvoie les métriques principales sous forme de dict plat.

    Précision / rappel / F1 sont en moyenne macro. ``zero_division=0`` garde le
    score défini si un modèle ne prédit jamais une classe.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    """Sauvegarde une matrice de confusion (lignes = vraies, colonnes = prédites).

    Les classes sont ordonnées selon :data:`config.CLASS_NAMES` pour que toutes
    les matrices du projet soient comparables.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(config.CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, metric: str, path: Path) -> None:
    """Diagramme en barres groupées d'une métrique par vectoriseur x classifieur.

    Attend une table avec les colonnes ``vectorizer``, ``model`` et ``<metric>``.
    """
    pivot = results.pivot(index="model", columns="vectorizer", values=metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Comparaison des modèles selon {metric}")
    ax.legend(title="Vectoriseur", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_bar(series: pd.Series, title: str, ylabel: str, path: Path, ylim=None) -> None:
    """Diagramme en barres d'une série (ablations, distributions)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    series.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_line(x, y, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    """Courbe simple (variance SVD, élagage, early stopping)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", markersize=3, color="darkorange")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
