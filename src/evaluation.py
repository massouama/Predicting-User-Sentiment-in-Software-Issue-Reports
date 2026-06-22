"""Metrics and visualisation helpers.

Centralises everything the brief lists under *Evaluation* -- accuracy,
precision, recall, F1 and the confusion matrix -- plus the comparison charts
used in the report.  Macro-averaging is used throughout because, on an
imbalanced problem, it weights every class equally and so reflects minority-class
performance (which plain accuracy hides).

A non-interactive Matplotlib backend is selected at import time so the module
renders figures to disk on a headless server without any display.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede the pyplot import
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
    """Return the headline metrics as a flat dict.

    Precision / recall / F1 are macro-averaged (unweighted mean over classes).
    ``zero_division=0`` keeps the score well-defined if a model never predicts
    some class.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, path: Path) -> None:
    """Save a labelled confusion-matrix heat-map.

    Rows are true classes, columns predicted, both ordered by
    :data:`config.CLASS_NAMES` so every matrix in the project is comparable.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(config.CLASS_NAMES))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, metric: str, path: Path) -> None:
    """Grouped bar chart of one metric across vectoriser x classifier combos.

    Expects a tidy frame with ``vectorizer``, ``model`` and ``<metric>`` columns
    (as produced by the experiment runner).
    """
    pivot = results.pivot(index="model", columns="vectorizer", values=metric)
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"Model comparison by {metric}")
    ax.legend(title="Vectoriser", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)


def plot_bar(series: pd.Series, title: str, ylabel: str, path: Path, ylim=None) -> None:
    """Generic single-series bar chart (used for ablations and distributions)."""
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
    """Generic line plot (PCA variance curve, pruning / early-stopping curves)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, marker="o", markersize=3, color="darkorange")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=config.FIG_DPI)
    plt.close(fig)
