"""evaluation.py — Métriques et figures.

Fonctions partagées par toutes les expériences pour :
  - calculer accuracy, précision, rappel, F1 macro
  - tracer et sauvegarder des matrices de confusion
  - tracer des graphiques de comparaison

On utilise systématiquement la macro-moyenne car le jeu de données est
déséquilibré : la macro-F1 donne le même poids à chaque classe,
peu importe sa taille (contrairement à l'accuracy ou la micro-moyenne).
"""
import matplotlib
matplotlib.use("Agg")   # rendu sans écran (serveur, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


def calculer_metriques(y_vrai, y_pred) -> dict:
    """Calcule et renvoie les quatre métriques principales.

    zero_division=0 : si un modèle ne prédit jamais une classe,
    on affiche 0 au lieu de déclencher une erreur ou un avertissement.
    """
    return {
        "accuracy":         accuracy_score(y_vrai, y_pred),
        "precision_macro":  precision_score(y_vrai, y_pred, average="macro", zero_division=0),
        "recall_macro":     recall_score(   y_vrai, y_pred, average="macro", zero_division=0),
        "f1_macro":         f1_score(       y_vrai, y_pred, average="macro", zero_division=0),
    }


def tracer_matrice_confusion(y_vrai, y_pred, titre: str, chemin: Path) -> None:
    """Sauvegarde une heatmap de la matrice de confusion.

    Lignes = classes réelles, colonnes = classes prédites.
    L'ordre des classes suit config.CLASSES pour être cohérent partout.
    """
    mc = confusion_matrix(y_vrai, y_pred, labels=list(config.CLASSES))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.CLASSES, yticklabels=config.CLASSES, ax=ax)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Réel")
    ax.set_title(titre)
    fig.tight_layout()
    fig.savefig(chemin, dpi=config.DPI_FIGURE)
    plt.close(fig)


def tracer_barres(serie: pd.Series, titre: str, ylabel: str, chemin: Path,
                  ylim=None) -> None:
    """Graphique en barres générique (distribution de classes, ablations…)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    serie.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(titre)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(chemin, dpi=config.DPI_FIGURE)
    plt.close(fig)


def tracer_comparaison(resultats: pd.DataFrame, metrique: str, chemin: Path) -> None:
    """Graphique en barres groupées : classifieurs × vectoriseurs pour une métrique."""
    pivot = resultats.pivot(index="modele", columns="vectoriseur", values=metrique)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metrique)
    ax.set_ylim(0, 1)
    ax.set_title(f"Comparaison des modèles — {metrique}")
    ax.legend(title="Vectoriseur", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(chemin, dpi=config.DPI_FIGURE)
    plt.close(fig)


def tracer_ligne(x, y, titre: str, xlabel: str, ylabel: str, chemin: Path) -> None:
    """Graphique en ligne générique (courbe de variance expliquée, courbe de pruning…)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", markersize=3, color="darkorange")
    ax.set_title(titre)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(chemin, dpi=config.DPI_FIGURE)
    plt.close(fig)
