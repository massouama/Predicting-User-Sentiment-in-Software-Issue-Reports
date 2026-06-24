"""config.py — Tous les réglages du projet en un seul endroit.

On évite les "nombres magiques" éparpillés dans le code :
tout ce qui peut changer est ici.
"""
from pathlib import Path

# ── Reproductibilité ─────────────────────────────────────────────────────────
# La même graine partout = exactement les mêmes résultats à chaque exécution.
GRAINE = 42

# ── Dossiers ─────────────────────────────────────────────────────────────────
RACINE        = Path(__file__).resolve().parent.parent   # racine du dépôt
DONNEES_DIR   = RACINE / "data"
RESULTATS_DIR = RACINE / "results"
FIGURES_DIR   = RESULTATS_DIR / "figures"
MATRICES_DIR  = RESULTATS_DIR / "confusion_matrices"
TABLES_DIR    = RESULTATS_DIR / "tables"

# Chemin du fichier d'avis Facebook (fourni dans le dépôt).
FACEBOOK_CSV  = DONNEES_DIR / "facebook_reviews.csv"

# ── Classes de sentiment ──────────────────────────────────────────────────────
# Correspondance étoiles → sentiment (1-2★ négatif, 3★ neutre, 4-5★ positif).
ETOILES_VERS_SENTIMENT = {1: "negative", 2: "negative", 3: "neutral",
                           4: "positive", 5: "positive"}
CLASSES = ("negative", "neutral", "positive")

# ── Découpage train / test ───────────────────────────────────────────────────
TAILLE_TEST = 0.20    # 20 % des données servent de test final

# ── Validation croisée ───────────────────────────────────────────────────────
NB_PLIS  = 5             # 5-fold StratifiedKFold
METRIQUE = "f1_macro"    # métrique de réglage (macro-F1 pénalise les classes ignorées)

# ── Vectorisation (BoW / TF-IDF) ────────────────────────────────────────────
MAX_MOTS    = 5000   # on garde les 5 000 mots les plus fréquents
MIN_DF      = 2      # on ignore les mots apparus dans un seul document
NGRAM       = (1, 2) # unigrammes + bigrammes

# ── Word2Vec (entraîné sur nos données) ──────────────────────────────────────
W2V_TAILLE  = 100    # dimension des vecteurs de mots
W2V_FENETRE = 5      # contexte de chaque mot (5 mots avant / après)
W2V_MIN     = 2      # on ignore les mots trop rares
W2V_EPOCHS  = 30     # nombre de passes d'entraînement

# ── Réduction de dimension (SVD / "PCA du texte") ────────────────────────────
SVD_COMPOSANTES = 300

# ── Figures ──────────────────────────────────────────────────────────────────
DPI_FIGURE = 120


def creer_dossiers():
    """Crée tous les dossiers de sortie s'ils n'existent pas encore."""
    for d in (DONNEES_DIR, RESULTATS_DIR, FIGURES_DIR, MATRICES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)
