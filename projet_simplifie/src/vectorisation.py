"""vectorisation.py — Conversion du texte en nombres.

Trois stratégies comparées dans le projet :
  - BoW  : compte le nombre d'occurrences de chaque mot (Bag of Words)
  - TF-IDF : pondère les mots rares plus fortement
  - Word2Vec : représente chaque mot par un vecteur dense (sémantique)

Chaque fabrique renvoie un objet sklearn-compatible (fit / transform).
"""
import numpy as np
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import config


# ── Bag of Words ──────────────────────────────────────────────────────────────
def creer_bow():
    """Compte de mots simples avec bigrammes.

    max_features=5000 : on garde les 5000 mots les plus fréquents.
    min_df=2 : on ignore les mots apparus dans un seul document (bruit).
    ngram_range=(1,2) : on compte aussi les paires de mots adjacents.
    """
    return CountVectorizer(
        max_features=config.MAX_MOTS,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM,
    )


# ── TF-IDF ────────────────────────────────────────────────────────────────────
def creer_tfidf():
    """TF-IDF avec log-fréquence (sublinear_tf=True atténue les mots très répétés)."""
    return TfidfVectorizer(
        max_features=config.MAX_MOTS,
        min_df=config.MIN_DF,
        ngram_range=config.NGRAM,
        sublinear_tf=True,
    )


# ── Word2Vec ──────────────────────────────────────────────────────────────────
class VectoriseurWord2Vec(BaseEstimator, TransformerMixin):
    """Entraîne Word2Vec sur les données d'entraînement et représente
    chaque document par la moyenne des vecteurs de ses mots.

    Pourquoi la moyenne ? Un vecteur de mots résume le sens moyen du texte.
    Les mots absents du vocabulaire sont simplement ignorés.
    """

    def fit(self, X, y=None):
        # Tokenise chaque document en liste de mots.
        phrases = [doc.split() for doc in X]
        # Entraîne Word2Vec sur le corpus.
        self.modele_ = Word2Vec(
            sentences=phrases,
            vector_size=config.W2V_TAILLE,
            window=config.W2V_FENETRE,
            min_count=config.W2V_MIN,
            epochs=config.W2V_EPOCHS,
            seed=config.GRAINE,
            workers=1,   # reproductible (pas de parallélisme aléatoire)
        )
        return self

    def transform(self, X):
        """Renvoie un tableau (n_docs × W2V_TAILLE) de vecteurs moyens."""
        vecteurs = []
        for doc in X:
            mots = [m for m in doc.split() if m in self.modele_.wv]
            if mots:
                vecteurs.append(self.modele_.wv[mots].mean(axis=0))
            else:
                # Document vide ou aucun mot connu → vecteur nul.
                vecteurs.append(np.zeros(config.W2V_TAILLE))
        return np.array(vecteurs)


# ── Registre des vectoriseurs ─────────────────────────────────────────────────
# Clé "dense" indique si la sortie est un tableau dense (True) ou
# une matrice creuse (False). Cela détermine quel Naive Bayes utiliser.
VECTORISEURS = {
    "BoW":      {"fabrique": creer_bow,           "dense": False},
    "TF-IDF":   {"fabrique": creer_tfidf,         "dense": False},
    "Word2Vec": {"fabrique": VectoriseurWord2Vec,  "dense": True},
}
