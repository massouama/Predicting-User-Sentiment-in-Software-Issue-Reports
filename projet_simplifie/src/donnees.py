"""donnees.py — Chargement et préparation du jeu de données Facebook.

On charge les avis Facebook (fichier CSV fourni dans data/),
on convertit les étoiles en sentiment, on déduplique les textes
puis on divise en train / test de façon stratifiée.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

import config


def charger_avis_facebook() -> pd.DataFrame:
    """Charge le fichier CSV Facebook et renvoie un DataFrame propre.

    Colonnes utiles : 'text' (contenu de l'avis) et 'label' (sentiment).
    On déduplique les textes identiques pour éviter que le même avis
    se retrouve à la fois dans le train et dans le test.
    """
    df = pd.read_csv(config.FACEBOOK_CSV)

    # Garde uniquement le texte et le score, renomme les colonnes.
    df = df[["content", "score"]].rename(columns={"content": "text", "score": "stars"})

    # Supprime les lignes sans texte.
    df = df.dropna(subset=["text"])

    # Convertit les étoiles (1-5) en classe de sentiment.
    df["label"] = df["stars"].map(config.ETOILES_VERS_SENTIMENT)
    df = df.dropna(subset=["label"])

    # Déduplique les textes : un texte identique dans train ET test serait
    # une fuite de données (le modèle aurait "déjà vu" ce texte).
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    return df[["text", "label"]]


def diviser_donnees(df: pd.DataFrame):
    """Divise le DataFrame en ensembles d'entraînement et de test.

    La stratification garantit que chaque ensemble respecte les proportions
    de classes du corpus entier (important car le neutre est très rare).

    Renvoie : X_train, X_test, y_train, y_test (tableaux numpy de strings).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].to_numpy(),
        df["label"].to_numpy(),
        test_size=config.TAILLE_TEST,
        stratify=df["label"],          # même répartition dans train et test
        random_state=config.GRAINE,
    )
    return X_train, X_test, y_train, y_test
