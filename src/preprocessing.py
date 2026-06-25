"""Nettoyage et normalisation du texte.

Pipeline NLP classique : *normaliser -> tokeniser -> retirer les mots vides ->
lemmatiser*, avec deux ajustements orientés sentiment :

1. **On garde les négations.** Les listes de mots vides retirent ``not``, ``no``,
   ``never``... or ces mots inversent le sentiment (« not working » ≠
   « working »), donc on les soustrait de l'ensemble des mots vides.
2. **Lemmatisation WordNet verbe puis nom.** Deux passes peu coûteuses
   (« running » -> « run », « issues » -> « issue ») donnent l'essentiel d'une
   lemmatisation POS-aware sans le coût d'un étiqueteur grammatical.

Tout est encapsulé dans :class:`TextPreprocessor`, un transformateur compatible
scikit-learn et sans état (``fit`` ne fait rien), qui s'insère directement dans
un :class:`~sklearn.pipeline.Pipeline`. Les ressources NLTK sont créées une
seule fois à l'import et partagées.
"""
from __future__ import annotations

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# --- Téléchargement NLTK (une seule fois) ----------------------------------
def _ensure_nltk_data() -> None:
    """Télécharge les petits corpus NLTK utilisés, seulement s'ils manquent."""
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

# Les négations doivent survivre au retrait des mots vides : elles portent le sentiment.
_NEGATION_WHITELIST: frozenset[str] = frozenset(
    {"no", "not", "nor", "never", "none", "nothing", "cannot", "without", "against"}
)
_STOPWORDS: frozenset[str] = frozenset(stopwords.words("english")) - _NEGATION_WHITELIST

_LEMMATIZER = WordNetLemmatizer()

# Expressions régulières compilées une fois, réutilisées pour chaque document.
_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")     # ne garde que lettres et espaces
_MULTISPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z]+")


def clean_text(text: str) -> str:
    """Met en minuscules et retire URLs, HTML et tout caractère non alphabétique."""
    text = str(text).lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()


def _lemmatize(token: str) -> str:
    """Lemmatisation WordNet en deux passes (forme verbale puis nominale)."""
    return _LEMMATIZER.lemmatize(_LEMMATIZER.lemmatize(token, pos="v"), pos="n")


def preprocess(text: str, min_token_len: int = 2) -> str:
    """Pipeline complet de normalisation pour un document.

    Étapes : :func:`clean_text` -> tokenisation -> retrait des mots vides et des
    tokens trop courts -> lemmatisation. Renvoie les tokens rejoints par des
    espaces, la forme attendue par les vectoriseurs scikit-learn.
    """
    cleaned = clean_text(text)
    tokens = _TOKEN_RE.findall(cleaned)
    return " ".join(
        _lemmatize(tok)
        for tok in tokens
        if len(tok) >= min_token_len and tok not in _STOPWORDS
    )


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """Transformateur scikit-learn sans état encapsulant :func:`preprocess`.

    En étant un vrai transformateur, le nettoyage peut vivre *dans* un pipeline,
    donc dans la validation croisée, éliminant tout risque de fuite train/test
    qu'aurait une étape de nettoyage appliquée manuellement à part.

    ``min_token_len`` (défaut 2) fixe la longueur minimale des tokens conservés
    (écarte les lettres isolées).
    """

    def __init__(self, min_token_len: int = 2):
        self.min_token_len = min_token_len

    def fit(self, X, y=None):  # noqa: N803
        """Sans effet : le nettoyage n'apprend rien des données."""
        return self

    def transform(self, X):  # noqa: N803
        """Nettoie chaque document de ``X`` -> liste de chaînes."""
        return [preprocess(doc, self.min_token_len) for doc in X]
