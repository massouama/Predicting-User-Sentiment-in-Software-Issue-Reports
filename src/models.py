"""Classifier definitions and their hyper-parameter search grids.

The brief asks for at least three classifiers plus tuning, pruning,
regularisation and (for boosting) early stopping.  We cover all of it:

* **Naive Bayes** -- ``MultinomialNB`` for sparse non-negative features
  (BoW / TF-IDF) and ``GaussianNB`` for dense embeddings, chosen automatically.
* **Decision Tree** -- with both *pre-pruning* (``max_depth``,
  ``min_samples_leaf``) and *post-pruning* (cost-complexity ``ccp_alpha``) in the
  grid.
* **Random Forest** -- bagged trees, a strong low-variance ensemble.
* **AdaBoost** -- boosted stumps, the required boosting method.
* **Logistic Regression** -- included specifically to demonstrate L2
  **regularisation** (the inverse-strength ``C`` is tuned).

Grid keys are prefixed with :data:`CLF_STEP` (``"clf"``) so they address the
classifier step of the surrounding pipeline directly.  Grids are deliberately
compact -- broad enough to show meaningful tuning, small enough to stay fast
under 5-fold cross-validation.

Early stopping is a property of a fitted estimator rather than a grid, so it is
demonstrated separately by :func:`build_early_stopping_model`.
"""
from __future__ import annotations

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import config

# Name of the classifier step inside every pipeline; grids reference it.
CLF_STEP = "clf"


def _prefixed(grid: dict) -> dict:
    """Prefix every grid key with ``"clf__"`` to target the pipeline step."""
    return {f"{CLF_STEP}__{k}": v for k, v in grid.items()}


def _naive_bayes(dense: bool):
    """Pick the Naive Bayes variant matching the feature space.

    ``MultinomialNB`` assumes non-negative counts and therefore only suits
    BoW / TF-IDF; dense embeddings (which contain negative values) require the
    Gaussian variant.
    """
    if dense:
        return GaussianNB(), _prefixed({"var_smoothing": [1e-9, 1e-8, 1e-7]})
    return MultinomialNB(), _prefixed({"alpha": [0.1, 0.5, 1.0]})


def build_classifiers(dense: bool = False) -> dict[str, dict]:
    """Return the model-comparison suite as ``{name: {estimator, param_grid}}``.

    Parameters
    ----------
    dense:
        ``True`` when the feature matrix is dense / may contain negatives
        (Word2Vec, BERT, or any SVD/PCA output).  Switches Naive Bayes to its
        Gaussian variant; every other classifier is feature-space agnostic.
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
                    # pre-pruning ...
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_leaf": [1, 5, 10],
                    # ... and post-pruning (cost-complexity)
                    "ccp_alpha": [0.0, 1e-3, 1e-2],
                }
            ),
        },
        "RandomForest": {
            # n_jobs left at 1: GridSearchCV parallelises folds, so keeping the
            # forest single-threaded avoids nested over-subscription of cores.
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
            # Tuning C demonstrates L2 regularisation strength selection.
            "param_grid": _prefixed({"C": [0.1, 1.0, 10.0]}),
        },
    }


def build_early_stopping_model(validation_fraction: float = 0.1):
    """Return a gradient-boosting classifier configured with early stopping.

    ``n_iter_no_change`` makes :class:`GradientBoostingClassifier` monitor a
    held-out validation slice and halt once the validation loss stops improving
    -- automatic early stopping that prevents over-fitting and wasted trees.
    Used by the dedicated early-stopping experiment.
    """
    return GradientBoostingClassifier(
        n_estimators=500,                 # generous ceiling ...
        learning_rate=0.1,
        validation_fraction=validation_fraction,
        n_iter_no_change=10,              # ... stop after 10 stagnant rounds
        random_state=config.RANDOM_STATE,
    )
