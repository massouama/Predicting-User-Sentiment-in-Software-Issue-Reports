"""Configuration centrale du projet de classification de sentiment.

Toutes les constantes ajustables sont regroupées ici pour éviter les « nombres
magiques » dans le reste du code. Les chemins sont résolus relativement à ce
fichier, ce qui rend le projet exécutable depuis n'importe quel répertoire.
"""
from __future__ import annotations

from pathlib import Path

# --- Reproductibilité ------------------------------------------------------
# Une graine globale unique, propagée au découpage train/test, au mélange de la
# validation croisée et à chaque estimateur stochastique : c'est ce qui rend les
# résultats reproductibles d'une exécution à l'autre.
RANDOM_STATE: int = 42

# --- Arborescence des fichiers ---------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
CONFUSION_DIR: Path = RESULTS_DIR / "confusion_matrices"
TABLES_DIR: Path = RESULTS_DIR / "tables"
MODELS_DIR: Path = RESULTS_DIR / "models"

# Corpus traité, mis en cache une fois puis réutilisé.
DATASET_PATH: Path = DATA_DIR / "issue_sentiment.csv"

# --- Jeu de données --------------------------------------------------------
# Avis de l'application Facebook (export Kaggle : colonne `content` + note
# `score` de 1 à 5), fournis dans le dépôt.
FACEBOOK_CSV_PATH: Path = DATA_DIR / "facebook_reviews.csv"
FACEBOOK_TEXT_COL: str = "content"
FACEBOOK_SCORE_COL: str = "score"

# Classes ordonnées du négatif au positif (ordre réutilisé pour toutes les
# matrices de confusion afin que les axes soient toujours cohérents).
CLASS_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")

# Conversion note 1-5 -> sentiment (1-2 = négatif, 3 = neutre, 4-5 = positif).
STAR_TO_SENTIMENT: dict[int, str] = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}

# --- Découpage train/test & validation croisée -----------------------------
TEST_SIZE: float = 0.20            # part réservée au test final
CV_FOLDS: int = 5                  # validation croisée à 5 plis (exigée par le sujet)
SCORING: str = "f1_macro"          # critère de réglage : équilibré entre classes

# --- Vectorisation (hyper-paramètres fixes ; les grilles des classifieurs
#     sont dans models.py) -------------------------------------------------
MAX_FEATURES: int = 5000           # taille max du vocabulaire (vitesse / mémoire)
MIN_DF: int = 2                    # ignore les mots présents dans un seul document
NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrammes + bigrammes

# Word2Vec entraîné sur le corpus d'entraînement.
W2V_VECTOR_SIZE: int = 100
W2V_WINDOW: int = 5
W2V_MIN_COUNT: int = 2
W2V_EPOCHS: int = 30

# TruncatedSVD (« PCA du texte » / LSA) : dimension cible.
SVD_COMPONENTS: int = 300

# --- Figures ---------------------------------------------------------------
FIG_DPI: int = 120


def ensure_directories() -> None:
    """Crée chaque dossier de sortie s'il n'existe pas déjà."""
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CONFUSION_DIR, TABLES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
