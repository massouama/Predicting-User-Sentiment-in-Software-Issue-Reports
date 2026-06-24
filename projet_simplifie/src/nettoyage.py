"""nettoyage.py — Nettoyage et normalisation du texte.

Pipeline classique NLP :
  1. Mise en minuscules + suppression des URLs, HTML, ponctuations
  2. Tokenisation (extraction des mots)
  3. Suppression des mots vides (stop-words), SAUF les négations
  4. Lemmatisation WordNet (deux passes : verbe puis nom)

Pourquoi garder les négations ?
  "not working" ≠ "working" → "not" renverse le sentiment,
  donc on ne le supprime pas.
"""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin


# ── Téléchargement NLTK (une seule fois) ─────────────────────────────────────
def _telecharger_nltk():
    for ressource, chemin in [
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ]:
        try:
            nltk.data.find(chemin)
        except LookupError:
            nltk.download(ressource, quiet=True)

_telecharger_nltk()

# ── Ressources partagées (créées une seule fois au chargement du module) ──────
# Mots vides anglais, MOINS les négations qui portent le sentiment.
_NEGATIONS = frozenset({"no", "not", "nor", "never", "none", "nothing",
                         "cannot", "without", "against"})
_MOTS_VIDES = frozenset(stopwords.words("english")) - _NEGATIONS

_LEMMATISEUR = WordNetLemmatizer()

# Expressions régulières compilées une fois pour traiter des milliers de textes.
_RE_URL       = re.compile(r"http\S+|www\.\S+")
_RE_HTML      = re.compile(r"<[^>]+>")
_RE_NON_ALPHA = re.compile(r"[^a-z\s]")
_RE_ESPACES   = re.compile(r"\s+")
_RE_MOT       = re.compile(r"[a-z]+")


def _nettoyer(texte: str) -> str:
    """Étape 1 : met en minuscules et supprime tout ce qui n'est pas une lettre."""
    texte = str(texte).lower()
    texte = _RE_URL.sub(" ", texte)
    texte = _RE_HTML.sub(" ", texte)
    texte = _RE_NON_ALPHA.sub(" ", texte)
    return _RE_ESPACES.sub(" ", texte).strip()


def _lemmatiser(mot: str) -> str:
    """Étape 4 : ramène le mot à sa forme de base (ex. "running" → "run")."""
    # Deux passes : forme verbale d'abord, puis forme nominale.
    return _LEMMATISEUR.lemmatize(
        _LEMMATISEUR.lemmatize(mot, pos="v"), pos="n"
    )


def pretraiter(texte: str, longueur_min: int = 2) -> str:
    """Applique tout le pipeline sur un seul texte et renvoie une chaîne."""
    nettoye = _nettoyer(texte)
    mots = _RE_MOT.findall(nettoye)
    return " ".join(
        _lemmatiser(m)
        for m in mots
        if len(m) >= longueur_min and m not in _MOTS_VIDES
    )


class Preprocesseur(BaseEstimator, TransformerMixin):
    """Transformateur scikit-learn qui applique `pretraiter` sur une liste de textes.

    Être un transformateur sklearn permet de l'intégrer dans un Pipeline,
    ce qui garantit qu'il s'applique séparément sur chaque pli de la CV
    (pas de fuite de données entre train et validation).
    """

    def fit(self, X, y=None):
        # Le nettoyage est stateless : il n'apprend rien des données.
        return self

    def transform(self, X):
        """Nettoie chaque texte de la liste X."""
        return [pretraiter(t) for t in X]
