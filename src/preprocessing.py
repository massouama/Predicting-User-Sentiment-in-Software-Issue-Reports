"""Text cleaning and normalisation for issue reports.

The pipeline is the classic NLP sequence -- *normalise -> tokenise -> remove
stop-words -> lemmatise* -- with two deliberate, sentiment-aware tweaks:

1. **Negation words are kept.**  Generic stop-word lists drop ``not``, ``no``,
   ``never`` ... but those flip sentiment ("not working" vs "working"), so we
   subtract them from the stop-word set.
2. **WordNet lemmatisation is applied verb-then-noun.**  A single cheap two-pass
   call ("running" -> "run", "issues" -> "issue") gives most of the benefit of
   full POS-aware lemmatisation without the cost of a POS tagger.

The work is encapsulated in :class:`TextPreprocessor`, a stateless,
scikit-learn-compatible transformer (``fit`` is a no-op) so it can drop straight
into a :class:`~sklearn.pipeline.Pipeline`.  Heavyweight NLTK resources
(lemmatiser, stop-word set, compiled regexes) are created **once** at module
import and shared, which keeps transformation fast over thousands of documents.
"""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# One-time NLTK resource bootstrap
# ---------------------------------------------------------------------------
def _ensure_nltk_data() -> None:
    """Download the small NLTK corpora we rely on, only if they are missing."""
    for resource, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()

# Negation cues must survive stop-word removal because they carry sentiment.
_NEGATION_WHITELIST: frozenset[str] = frozenset(
    {"no", "not", "nor", "never", "none", "nothing", "cannot", "without", "against"}
)
_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english")) - _NEGATION_WHITELIST

_LEMMATIZER = WordNetLemmatizer()

# Pre-compiled regexes (compiled once, reused for every document).
_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")     # keep letters and whitespace only
_MULTISPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+")           # whitespace-free word tokeniser


def clean_text(text: str) -> str:
    """Lower-case and strip URLs, HTML and every non-alphabetic character.

    Returns a normalised, single-spaced string ready for tokenisation.
    """
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def _lemmatize(token: str) -> str:
    """Two-pass WordNet lemmatisation (verb form first, then noun form)."""
    return _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(token, pos="v"), pos="n")


def preprocess(text: str, min_token_len: int = 2) -> str:
    """Full normalisation pipeline for a single document.

    Steps: :func:`clean_text` -> regex tokenise -> drop stop-words and very
    short tokens -> lemmatise.  Returns the processed tokens re-joined into a
    space-separated string, the form expected by scikit-learn vectorisers.
    """
    cleaned = clean_text(text)
    tokens = _TOKEN_RE.findall(cleaned)
    return " ".join(
        _lemmatize(tok)
        for tok in tokens
        if len(tok) >= min_token_len and tok not in _STOPWORDS
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Stateless scikit-learn transformer wrapping :func:`preprocess`.

    Being a proper transformer means preprocessing can live *inside* a pipeline
    and therefore inside cross-validation, eliminating any risk of train/test
    leakage from a separate, manually applied cleaning step.

    Parameters
    ----------
    min_token_len:
        Minimum surviving token length (default 2 drops stray single letters).
    """

    def __init__(self, min_token_len: int = 2):
        self.min_token_len = min_token_len

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming convention
        """No-op: preprocessing learns nothing from the data."""
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming convention
        """Clean every document in the iterable ``X`` -> list of strings."""
        return [preprocess(doc, self.min_token_len) for doc in X]
