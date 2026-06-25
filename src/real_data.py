"""Chargement du corpus d'avis Facebook étiqueté en sentiment.

Le jeu de données est un export Kaggle d'avis de l'application Facebook
(colonne ``content`` + une note ``score`` de 1 à 5). On convertit la note en
sentiment, on supprime les textes vides et les doublons (un texte identique
présent à la fois dans le train et le test fausserait l'évaluation), puis on
mélange le tout de façon reproductible.

Le corpus traité est mis en cache dans :data:`config.DATASET_PATH` afin que
toutes les étapes en aval travaillent exactement sur les mêmes données.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def build_facebook_reviews_dataset(
    path=config.FACEBOOK_CSV_PATH,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Construit le corpus étiqueté ``text,label`` à partir du CSV Facebook."""
    raw = pd.read_csv(path, usecols=[config.FACEBOOK_TEXT_COL, config.FACEBOOK_SCORE_COL])
    out = pd.DataFrame(
        {
            "text": raw[config.FACEBOOK_TEXT_COL].astype(str).str.strip(),
            "label": raw[config.FACEBOOK_SCORE_COL].map(config.STAR_TO_SENTIMENT),
        }
    )
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0].drop_duplicates(subset="text")
    # Mélange reproductible : les classes sont entrelacées avant tout découpage.
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def load_or_create_dataset(path=config.DATASET_PATH) -> pd.DataFrame:
    """Renvoie le corpus mis en cache, en le construisant au premier appel.

    Mettre le CSV traité en cache garantit que toutes les étapes (exploration,
    entraînement, rapport) utilisent exactement les mêmes données et rend les
    exécutions reproductibles.
    """
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)

    config.ensure_directories()
    df = build_facebook_reviews_dataset()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    data = load_or_create_dataset()
    print(f"{len(data)} avis Facebook étiquetés")
    print("\nRépartition des classes :")
    print(data["label"].value_counts().to_string())
