"""Gestion du déséquilibre de classes du corpus.

Les avis sont fortement déséquilibrés (les positifs dominent, le neutre est
minuscule). Sans traitement, les classifieurs optimisent l'accuracy globale en
négligeant les classes minoritaires. On compare deux remèdes standards :

* **SMOTE** : synthétise de nouveaux exemples minoritaires en interpolant entre
  points voisins, enrichissant la frontière de décision sans dupliquer de lignes.
* **Sous-échantillonnage aléatoire** : retire des exemples majoritaires jusqu'à
  équilibrer les classes ; rapide, mais jette de la donnée.

Le rééchantillonnage ne doit agir **que sur les plis d'entraînement**. On
renvoie donc des *samplers* :mod:`imblearn` qui s'insèrent dans un
:class:`imblearn.pipeline.Pipeline`, où ils sont automatiquement ignorés à la
prédiction — la seule façon robuste d'éviter de fuiter des exemples synthétiques
dans l'évaluation.
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import config


def make_smote(k_neighbors: int = 5) -> SMOTE:
    """Sur-échantillonneur SMOTE équilibrant chaque classe sur la majoritaire.

    ``k_neighbors`` est le nombre de voisins minoritaires servant à interpoler
    chaque point synthétique ; il doit rester sous la taille de la plus petite
    classe.
    """
    return SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)


def make_undersampler() -> RandomUnderSampler:
    """Sous-échantillonneur aléatoire ramenant chaque classe à la plus petite."""
    return RandomUnderSampler(random_state=config.RANDOM_STATE)


# Registre consommé par l'expérience sur le déséquilibre. ``None`` est la
# baseline non traitée servant de comparaison.
SAMPLERS: dict[str, object] = {
    "none": None,
    "SMOTE": make_smote(),
    "undersample": make_undersampler(),
}


def class_distribution(labels) -> pd.Series:
    """Renvoie les effectifs par classe, ordonnés selon :data:`config.CLASS_NAMES`."""
    counts = pd.Series(labels).value_counts()
    return counts.reindex(config.CLASS_NAMES).fillna(0).astype(int)
