"""Stratégies de vectorisation du texte comparées dans le projet.

Quatre représentations, du comptage creux aux plongements denses :

    BoW                   comptes bruts de n-grammes            creux, >= 0
    TF-IDF                comptes pondérés par la rareté         creux, >= 0
    Word2Vec              moyenne de plongements (appris ici)    dense, +/-
    Word2Vec pré-entraîné moyenne des vecteurs Google News       dense, +/-

Chaque stratégie est exposée via une fabrique ``make_*`` qui renvoie un
estimateur compatible scikit-learn. La sortie creuse et positive de BoW/TF-IDF
les rend compatibles avec ``MultinomialNB`` ; les plongements denses (qui ont
des valeurs négatives) non, ce qui est géré dans :mod:`src.models`.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config


# --- Vectoriseurs creux par comptage (intégrés à scikit-learn) -------------
def make_bow() -> CountVectorizer:
    """Bag-of-Words : comptes bruts d'occurrences de n-grammes.

    La représentation la plus simple ; une baseline rapide et solide, et
    l'entrée naturelle du Naive Bayes multinomial.
    """
    return CountVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM_RANGE,
    )


def make_tfidf() -> TfidfVectorizer:
    """TF-IDF : comptes atténués selon la fréquence d'un terme dans le corpus.

    Les mots rares et discriminants sont privilégiés, ce qui bat généralement le
    simple BoW sur des textes courts et bruités.
    """
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True,   # 1 + log(tf) : atténue les termes répétés
    )


# --- Vectoriseur Word2Vec (plongements appris sur le corpus) ---------------
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Plongements de document Word2Vec moyennés.

    Un modèle Word2Vec est entraîné sur le corpus d'entraînement dans ``fit``
    (aucune fuite du test), puis chaque document est représenté par la moyenne
    des vecteurs de ses mots connus. Les documents vides donnent le vecteur nul.
    """

    def __init__(
        self,
        vector_size: int = config.W2V_VECTOR_SIZE,
        window: int = config.W2V_WINDOW,
        min_count: int = config.W2V_MIN_COUNT,
        epochs: int = config.W2V_EPOCHS,
        random_state: int = config.RANDOM_STATE,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.random_state = random_state

    @staticmethod
    def _tokenize(documents):
        """Découpe les chaînes déjà nettoyées en listes de tokens pour gensim."""
        return [doc.split() for doc in documents]

    def fit(self, X, y=None):  # noqa: N803
        """Entraîne Word2Vec sur les documents d'entraînement tokenisés."""
        from gensim.models import Word2Vec  # import local : gensim reste optionnel

        self.model_ = Word2Vec(
            sentences=self._tokenize(X),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.random_state,
            workers=1,  # un seul worker -> vecteurs déterministes et reproductibles
        )
        return self

    def transform(self, X):  # noqa: N803
        """Moyenne les vecteurs de mots en un vecteur dense par document."""
        wv = self.model_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, doc in enumerate(self._tokenize(X)):
            vectors = [wv[tok] for tok in doc if tok in wv.key_to_index]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out


def make_word2vec() -> Word2VecVectorizer:
    """Fabrique de :class:`Word2VecVectorizer` avec les réglages par défaut."""
    return Word2VecVectorizer()


# --- Vectoriseur Word2Vec pré-entraîné (vecteurs Google News via gensim) ---
# Cache des vecteurs pré-entraînés au niveau module. Les vecteurs vivent ici,
# PAS sur l'instance, pour que le transformateur ajusté reste léger et ne soit
# jamais sérialisé dans le cache joblib du pipeline.
_PRETRAINED_CACHE: dict = {}

# Vecteurs Google News 300-d. Le modèle complet pèse ~3,6 Go (3 M de mots) ; on
# ne charge que les ``_PRETRAINED_LIMIT`` mots les plus fréquents (~0,6 Go),
# couverture amplement suffisante pour des avis courts.
_PRETRAINED_MODEL = "word2vec-google-news-300"
_PRETRAINED_LIMIT = 500_000


def pretrained_available(model_name: str = _PRETRAINED_MODEL) -> bool:
    """Renvoie ``True`` si les vecteurs Word2Vec pré-entraînés sont chargeables."""
    try:
        _load_pretrained(model_name)
        return True
    except Exception:
        return False


def _load_pretrained(model_name: str = _PRETRAINED_MODEL):
    """Charge (et met en cache) les ``KeyedVectors`` pré-entraînés."""
    if model_name not in _PRETRAINED_CACHE:
        import gensim.downloader as api
        from gensim.models import KeyedVectors

        path = api.load(model_name, return_path=True)
        _PRETRAINED_CACHE[model_name] = KeyedVectors.load_word2vec_format(
            path, binary=True, limit=_PRETRAINED_LIMIT
        )
    return _PRETRAINED_CACHE[model_name]


class PretrainedEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    """Plongements de document moyennés à partir de **Word2Vec pré-entraîné**.

    Contrairement à :class:`Word2VecVectorizer` (entraîné sur notre corpus), les
    plongements sont ici les vecteurs Google News, simplement chargés : ``fit``
    n'apprend rien. Chaque document devient la moyenne des vecteurs de ses mots
    connus ; les documents vides donnent le vecteur nul.
    """

    def __init__(self, model_name: str = _PRETRAINED_MODEL):
        self.model_name = model_name

    def fit(self, X, y=None):  # noqa: N803
        """Charge les vecteurs pré-entraînés et mémorise leur dimension."""
        kv = _load_pretrained(self.model_name)
        self.vector_size_ = kv.vector_size
        return self

    def transform(self, X):  # noqa: N803
        """Moyenne les vecteurs pré-entraînés en un vecteur dense par document."""
        kv = _load_pretrained(self.model_name)
        out = np.zeros((len(X), kv.vector_size), dtype=np.float32)
        for i, doc in enumerate(X):
            vectors = [kv[tok] for tok in doc.split() if tok in kv.key_to_index]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out


def make_pretrained() -> PretrainedEmbeddingVectorizer:
    """Fabrique de :class:`PretrainedEmbeddingVectorizer` (Word2Vec pré-entraîné)."""
    return PretrainedEmbeddingVectorizer()


# Registre consommé par l'orchestrateur. ``dense`` indique si la sortie peut
# contenir des valeurs négatives (donc incompatible avec MultinomialNB).
VECTORIZERS: dict[str, dict] = {
    "BoW": {"factory": make_bow, "dense": False},
    "TF-IDF": {"factory": make_tfidf, "dense": False},
    "Word2Vec": {"factory": make_word2vec, "dense": True},          # appris sur notre corpus
    "Word2Vec-pretrained": {"factory": make_pretrained, "dense": True},  # Google News
}
