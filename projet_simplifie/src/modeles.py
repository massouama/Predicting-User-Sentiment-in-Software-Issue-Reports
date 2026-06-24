"""modeles.py — Classifieurs et grilles d'hyperparamètres.

Le projet compare cinq classifieurs :
  - Naive Bayes (MultinomialNB pour les matrices creuses, GaussianNB pour les vecteurs denses)
  - Arbre de décision (avec pré- et post-élagage dans la grille)
  - Forêt aléatoire
  - AdaBoost (boosting)
  - Régression logistique (avec régularisation L2 réglée par C)

Les clés des grilles sont préfixées "clf__" pour pointer vers
l'étape "clf" dans le Pipeline sklearn.
"""
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import config


def _prefixer(grille: dict) -> dict:
    """Ajoute le préfixe 'clf__' à chaque clé de la grille."""
    return {f"clf__{k}": v for k, v in grille.items()}


def construire_classifieurs(dense: bool = False) -> dict:
    """Renvoie tous les classifieurs avec leur grille d'hyperparamètres.

    Paramètre `dense` :
      - False → features creuses (BoW / TF-IDF) → MultinomialNB
      - True  → features denses  (Word2Vec)     → GaussianNB
    """
    g = config.GRAINE

    # Choix automatique du bon Naive Bayes selon le type de features.
    if dense:
        # GaussianNB pour les vecteurs continus (Word2Vec peut avoir des valeurs négatives).
        nb      = GaussianNB()
        grille_nb = _prefixer({"var_smoothing": [1e-9, 1e-8, 1e-7]})
    else:
        # MultinomialNB pour les comptes de mots (toujours ≥ 0).
        # alpha = lissage de Laplace (évite P=0 pour des mots jamais vus à l'entraînement).
        nb      = MultinomialNB()
        grille_nb = _prefixer({"alpha": [0.1, 0.5, 1.0]})

    return {
        "NaiveBayes": {
            "estimateur": nb,
            "grille": grille_nb,
        },
        "DecisionTree": {
            "estimateur": DecisionTreeClassifier(random_state=g),
            "grille": _prefixer({
                # Pré-élagage : on limite la croissance de l'arbre.
                "max_depth":        [None, 10, 20, 30],   # None = pas de limite
                "min_samples_leaf": [1, 5, 10],           # taille minimale d'une feuille
                # Post-élagage : on coupe les branches inutiles après coup.
                "ccp_alpha":        [0.0, 1e-3, 1e-2],   # 0 = pas d'élagage
            }),
        },
        "RandomForest": {
            "estimateur": RandomForestClassifier(random_state=g),
            "grille": _prefixer({
                "n_estimators":     [200],           # nombre d'arbres
                "max_depth":        [None, 20],
                "min_samples_leaf": [1, 2],
                "max_features":     ["sqrt"],        # nb de features testées à chaque nœud
            }),
        },
        "AdaBoost": {
            "estimateur": AdaBoostClassifier(random_state=g),
            "grille": _prefixer({
                "n_estimators":  [100, 200],   # nombre d'itérations de boosting
                "learning_rate": [0.5, 1.0],  # pas d'apprentissage (grand = apprend vite mais risqué)
            }),
        },
        "LogisticRegression": {
            "estimateur": LogisticRegression(max_iter=1000, random_state=g),
            "grille": _prefixer({
                # C contrôle la régularisation L2 :
                #   petit C → régularise fort (modèle simple, moins de sur-apprentissage)
                #   grand C → régularise peu  (modèle colle aux données)
                "C": [0.1, 1.0, 10.0],
            }),
        },
    }


def construire_early_stopping() -> GradientBoostingClassifier:
    """Renvoie un Gradient Boosting avec arrêt automatique (early stopping).

    n_iter_no_change=10 : si le score de validation ne s'améliore pas
    pendant 10 tours consécutifs, on s'arrête (même si on n'a pas atteint
    les 500 arbres). L'arbre choisit lui-même son nombre optimal d'itérations.
    """
    return GradientBoostingClassifier(
        n_estimators=500,          # plafond généreux
        learning_rate=0.1,
        validation_fraction=0.1,   # 10 % du train surveillé pour l'arrêt
        n_iter_no_change=10,       # patience : 10 tours sans amélioration → stop
        random_state=config.GRAINE,
    )
