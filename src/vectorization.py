"""Text vectorisation strategies compared in this project.

Four families are provided, covering the spectrum from sparse count statistics
to dense contextual embeddings:

============  ====================================================  ===========
Vectoriser    Idea                                                  Output
============  ====================================================  ===========
Bag-of-Words  raw n-gram counts                                     sparse, >=0
TF-IDF        counts re-weighted by inverse document frequency      sparse, >=0
Word2Vec      mean of trained word embeddings (one vector / doc)    dense, +/-
BERT          mean-pooled contextual embeddings (optional)          dense, +/-
============  ====================================================  ===========

Each strategy is exposed through a ``make_*`` factory returning a fresh,
scikit-learn-compatible estimator, so callers can freely compose them into
pipelines.  The sparse, non-negative output of BoW/TF-IDF is what makes them
compatible with ``MultinomialNB``; the dense embedding outputs are not, which is
handled centrally in :mod:`src.models`.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config


# ---------------------------------------------------------------------------
# Sparse, count-based vectorisers (scikit-learn built-ins)
# ---------------------------------------------------------------------------
def make_bow() -> CountVectorizer:
    """Bag-of-Words: raw n-gram occurrence counts.

    The simplest representation -- a fast, strong baseline that is also the most
    natural input for Multinomial Naive Bayes.
    """
    return CountVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM_RANGE,
    )


def make_tfidf() -> TfidfVectorizer:
    """TF-IDF: counts down-weighted by how common a term is across documents.

    Rare, discriminative words are emphasised over ubiquitous ones, which
    usually beats plain BoW for short, noisy texts such as issue titles.
    """
    return TfidfVectorizer(
        max_features=config.MAX_FEATURES,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM_RANGE,
        sublinear_tf=True,   # 1 + log(tf): dampens the effect of repeated terms
    )


# ---------------------------------------------------------------------------
# Word2Vec embedding vectoriser
# ---------------------------------------------------------------------------
class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Average-pooled Word2Vec document embeddings.

    A Word2Vec model is trained on the *training* corpus during ``fit`` (so no
    test information leaks in), then each document is represented by the mean of
    its in-vocabulary word vectors -- a simple, robust sentence embedding.
    Out-of-vocabulary or empty documents map to the zero vector.

    Training embeddings on our own corpus -- rather than loading multi-gigabyte
    pre-trained vectors that cannot be fetched in this sandbox -- keeps the
    project self-contained while still demonstrating the embedding approach.

    Parameters mirror the ``config`` defaults and are exposed for tuning.
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
        """Split already-cleaned strings into token lists for gensim."""
        return [doc.split() for doc in documents]

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming convention
        """Train Word2Vec on the tokenised training documents."""
        from gensim.models import Word2Vec  # local import keeps gensim optional

        self.model_ = Word2Vec(
            sentences=self._tokenize(X),
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.random_state,
            workers=1,  # single worker -> deterministic, reproducible vectors
        )
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming convention
        """Mean-pool word vectors into one dense vector per document."""
        wv = self.model_.wv
        out = np.zeros((len(X), self.vector_size), dtype=np.float32)
        for i, doc in enumerate(self._tokenize(X)):
            vectors = [wv[tok] for tok in doc if tok in wv.key_to_index]
            if vectors:
                out[i] = np.mean(vectors, axis=0)
        return out


def make_word2vec() -> Word2VecVectorizer:
    """Factory for :class:`Word2VecVectorizer` using the configured defaults."""
    return Word2VecVectorizer()


# ---------------------------------------------------------------------------
# BERT embedding vectoriser (optional)
# ---------------------------------------------------------------------------
def bert_available() -> bool:
    """Return ``True`` only if both ``transformers`` and ``torch`` import."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


class BertVectorizer(BaseEstimator, TransformerMixin):
    """Mean-pooled contextual embeddings from a pre-trained BERT model.

    This is a complete, documented implementation of the BERT vectorisation
    branch required by the brief.  It is *optional*: ``transformers`` + ``torch``
    plus the model weights must be available locally.  In a network-restricted
    environment where the model hub is unreachable the weights cannot be
    downloaded, so the experiments fall back to BoW / TF-IDF / Word2Vec; the
    code path itself is nonetheless ready to run wherever BERT is installed.

    Each document is encoded and represented by the mean of its last-hidden-state
    token embeddings (attention-mask aware), a standard sentence-embedding
    recipe that is more robust than the raw ``[CLS]`` vector.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased", batch_size: int = 32, max_length: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming convention
        """Load the tokenizer and frozen model (no training takes place)."""
        if not bert_available():
            raise ImportError(
                "BertVectorizer requires the 'transformers' and 'torch' packages. "
                "Install them (and ensure the model weights are reachable) to use this vectoriser."
            )
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModel.from_pretrained(self.model_name)
        self.model_.eval()  # inference mode: embeddings only, weights frozen
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming convention
        """Encode documents in batches into mean-pooled embedding vectors."""
        torch = self._torch
        embeddings = []
        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                batch = list(X[start : start + self.batch_size])
                enc = self.tokenizer_(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                )
                hidden = self.model_(**enc).last_hidden_state          # (B, T, H)
                mask = enc["attention_mask"].unsqueeze(-1).float()      # (B, T, 1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                embeddings.append(pooled.cpu().numpy())
        return np.vstack(embeddings)


def make_bert() -> BertVectorizer:
    """Factory for :class:`BertVectorizer` (requires optional dependencies)."""
    return BertVectorizer()


# Registry consumed by the experiment runner.  ``dense`` flags whether the
# output contains negative values (and is therefore incompatible with
# MultinomialNB / direct PCA-on-counts), which downstream code keys off.
VECTORIZERS: dict[str, dict] = {
    "BoW": {"factory": make_bow, "dense": False},
    "TF-IDF": {"factory": make_tfidf, "dense": False},
    "Word2Vec": {"factory": make_word2vec, "dense": True},
    "BERT": {"factory": make_bert, "dense": True},
}
