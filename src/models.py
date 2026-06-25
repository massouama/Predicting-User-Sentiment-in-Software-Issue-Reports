"""Définitions des classifieurs et grilles d'hyper-paramètres.

Le sujet demande au moins trois classifieurs, plus réglage, élagage,
régularisation et (pour le boosting) early stopping. On couvre tout :

* **Naive Bayes** : ``MultinomialNB`` pour les features creuses positives
  (BoW / TF-IDF), ``GaussianNB`` pour les plongements denses, choisi
  automatiquement.
* **Decision Tree** : avec pré-élagage (``max_depth``, ``min_samples_leaf``) et
  post-élagage (``ccp_alpha``) dans la grille.
* **Random Forest** : ensemble d'arbres baggés, faible variance.
* **AdaBoost** : boosting de souches, la méthode de boosting demandée.
* **Logistic Regression** : démontre la régularisation L2 (``C`` réglé).

Les clés des grilles sont préfixées par :data:`CLF_STEP` (``"clf"``) pour cibler
l'étape classifieur du pipeline. Les grilles sont volontairement compactes :
assez larges pour montrer un vrai réglage, assez petites pour rester rapides en
validation croisée à 5 plis.
"""
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import config

# Nom de l'étape classifieur dans chaque pipeline ; les grilles le référencent.
CLF_STEP = "clf"


def _prefixed(grid: dict) -> dict:
    """Préfixe chaque clé de grille par ``"clf__"`` pour cibler l'étape pipeline."""
    return {f"{CLF_STEP}__{k}": v for k, v in grid.items()}


def _naive_bayes(dense: bool):
    """Choisit la variante de Naive Bayes adaptée à l'espace de features.

    ``MultinomialNB`` suppose des comptes positifs (BoW / TF-IDF) ; les
    plongements denses (qui ont des valeurs négatives) imposent la variante
    gaussienne.
    """
    if dense:
        return GaussianNB(), _prefixed({"var_smoothing": [1e-9, 1e-8, 1e-7]})
    return MultinomialNB(), _prefixed({"alpha": [0.1, 0.5, 1.0]})


def build_classifiers(dense: bool = False) -> dict[str, dict]:
    """Renvoie la suite de modèles ``{nom: {estimator, param_grid}}``.

    ``dense=True`` quand la matrice de features est dense / peut contenir des
    négatifs (Word2Vec ou sortie SVD) : bascule Naive Bayes en gaussien ; les
    autres classifieurs sont indifférents à l'espace de features.
    """
    rs = config.RANDOM_STATE
    nb_estimator, nb_grid = _naive_bayes(dense)

    return {
        "NaiveBayes": {
            "estimator": nb_estimator,
            "param_grid": nb_grid,
        },
        "DecisionTree": {
            "estimator": DecisionTreeClassifier(random_state=rs),
            "param_grid": _prefixed(
                {
                    # pré-élagage ...
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_leaf": [1, 5, 10],
                    # ... et post-élagage (cost-complexity)
                    "ccp_alpha": [0.0, 1e-3, 1e-2],
                }
            ),
        },
        "RandomForest": {
            # n_jobs reste à 1 : GridSearchCV parallélise déjà les plis, donc
            # garder la forêt mono-thread évite la sur-souscription des cœurs.
            "estimator": RandomForestClassifier(random_state=rs),
            "param_grid": _prefixed(
                {
                    "n_estimators": [200],
                    "max_depth": [None, 20],
                    "min_samples_leaf": [1, 2],
                    "max_features": ["sqrt"],
                }
            ),
        },
        "AdaBoost": {
            "estimator": AdaBoostClassifier(random_state=rs),
            "param_grid": _prefixed(
                {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.5, 1.0],
                }
            ),
        },
        "LogisticRegression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=rs),
            # Régler C illustre le choix de la force de régularisation L2.
            "param_grid": _prefixed({"C": [0.1, 1.0, 10.0]}),
        },
    }


def build_early_stopping_model(validation_fraction: float = 0.1):
    """Renvoie un gradient boosting configuré avec early stopping.

    ``n_iter_no_change`` fait surveiller à :class:`GradientBoostingClassifier`
    une part de validation et arrête l'entraînement dès que la perte de
    validation cesse de s'améliorer — early stopping automatique qui évite le
    surapprentissage et les arbres inutiles.
    """
    return GradientBoostingClassifier(
        n_estimators=500,                 # plafond généreux ...
        learning_rate=0.1,
        validation_fraction=validation_fraction,
        n_iter_no_change=10,              # ... arrêt après 10 tours sans progrès
        random_state=config.RANDOM_STATE,
    )
