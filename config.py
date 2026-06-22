"""Central configuration for the sentiment-classification project.

Every tunable constant lives here so that the rest of the code base stays free
of "magic numbers".  Paths are resolved relative to this file, which makes the
project runnable from any working directory.

The configuration is intentionally flat and declarative: importing ``config``
gives immediate, read-only access to paths, the random seed, dataset options
and the experiment grid, without executing any side effects.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A single global seed is threaded through data generation, the train/test
# split, cross-validation shuffling and every stochastic estimator.  Fixing it
# is what makes the reported numbers reproducible run after run.
RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT_DIR / "data"
RESULTS_DIR: Path = ROOT_DIR / "results"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
CONFUSION_DIR: Path = RESULTS_DIR / "confusion_matrices"
TABLES_DIR: Path = RESULTS_DIR / "tables"
MODELS_DIR: Path = RESULTS_DIR / "models"

# Canonical dataset location (generated once, then reused).
DATASET_PATH: Path = DATA_DIR / "issue_sentiment.csv"

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
# Which corpus to use:
#   "app_reviews" -> a REAL, pre-cleaned Google Play review dataset (288 k rows)
#                    from the open `sealuzh/user_quality` research repository,
#                    with star ratings mapped to sentiment.  This is the default
#                    and matches the brief's "App Store reviews dataset" example.
#   "synthetic"   -> the bundled reproducible generator (offline fallback).
DATASET_SOURCE: str = "app_reviews"

# Class labels, ordered from negative to positive.  This ordering is reused for
# every confusion matrix and report so axes are always consistent.
CLASS_NAMES: tuple[str, ...] = ("negative", "neutral", "positive")

# --- Real dataset (app reviews) -------------------------------------------
# Direct, credential-free download of the raw reviews CSV (id, package_name,
# review, date, star, version_id).
REVIEWS_URL: str = (
    "https://raw.githubusercontent.com/sealuzh/user_quality/master/csv_files/reviews.csv"
)
RAW_REVIEWS_PATH: Path = DATA_DIR / "reviews_raw.csv"   # cached download (git-ignored)
# Map 1-5 star ratings onto the three sentiment classes (the standard convention:
# 1-2 = negative, 3 = neutral, 4-5 = positive).
STAR_TO_SENTIMENT: dict[int, str] = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
# Stratified sample size kept (full 288 k is far larger than needed and slow);
# proportions of the natural, positive-heavy imbalance are preserved.
N_SAMPLES_REAL: int = 6000

# Number of synthetic issue reports to generate and the (deliberately skewed)
# class proportions.  Issue trackers are dominated by complaints, so negatives
# outnumber the rest -- this realistic imbalance is what the SMOTE / under-
# sampling experiments later address.
N_SAMPLES: int = 3000
CLASS_BALANCE: dict[str, float] = {"negative": 0.55, "neutral": 0.30, "positive": 0.15}

# Fraction of labels flipped to a random class, emulating annotation noise.
# Keeps the task non-trivial (caps the achievable accuracy below 100 %).
LABEL_NOISE: float = 0.08

# Realism / difficulty knobs.  Without them the sentiment cues are so clean that
# every classifier saturates at the same score and resampling has nothing to fix.
# These two probabilities inject the kind of ambiguity seen in real trackers:
#   * AMBIGUITY_PROB -- chance the core sentence uses a tone-neutral clause that
#     fits any class, so the label must be inferred from weaker contextual cues.
#   * MIXING_PROB    -- chance the opener/closing remark is borrowed from another
#     class, producing genuinely mixed-signal reports.
# Together they lower the ceiling to a realistic ~0.85-0.90 and, because they hit
# the rare positive class hardest, give SMOTE / under-sampling something to do.
AMBIGUITY_PROB: float = 0.22
MIXING_PROB: float = 0.15

# ---------------------------------------------------------------------------
# Train / test split & cross-validation
# ---------------------------------------------------------------------------
TEST_SIZE: float = 0.20            # held-out evaluation set
CV_FOLDS: int = 5                  # >= 5-fold as required by the brief
SCORING: str = "f1_macro"          # tuning target: balanced across classes

# ---------------------------------------------------------------------------
# Vectorisation hyper-parameters (fixed; classifier grids live in models.py)
# ---------------------------------------------------------------------------
# Shared vocabulary constraints for Bag-of-Words / TF-IDF.
MAX_FEATURES: int = 5000           # cap vocabulary for speed and memory
MIN_DF: int = 2                    # drop terms appearing in a single document
NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrams + bigrams capture short cues

# Word2Vec embedding settings (trained on the training corpus).
W2V_VECTOR_SIZE: int = 100
W2V_WINDOW: int = 5
W2V_MIN_COUNT: int = 2
W2V_EPOCHS: int = 30

# TruncatedSVD ("PCA for sparse text" / LSA) target dimensionality.
SVD_COMPONENTS: int = 300

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
FIG_DPI: int = 120
FIG_FORMAT: str = "png"


def ensure_directories() -> None:
    """Create every output directory if it does not already exist.

    Called once at the start of any script that writes artefacts so the rest of
    the code can assume the folders are present.
    """
    for directory in (DATA_DIR, RESULTS_DIR, FIGURES_DIR, CONFUSION_DIR, TABLES_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
