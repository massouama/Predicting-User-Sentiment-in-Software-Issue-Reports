"""Class-imbalance handling for the (deliberately skewed) issue corpus.

Issue trackers are dominated by complaints, so the ``negative`` class greatly
outnumbers ``positive``.  Left untreated, classifiers optimise overall accuracy
by neglecting the minority classes.  We compare two standard remedies:

* **SMOTE** (Synthetic Minority Over-sampling TEchnique) -- synthesises new
  minority examples by interpolating between neighbouring minority points,
  enriching the decision boundary without duplicating rows.
* **Random under-sampling** -- discards majority examples until the classes
  match; cheap and fast, but throws away data.

Crucially, resampling must run **only on the training folds**.  We therefore
return :mod:`imblearn` *samplers* that slot into an
:class:`imblearn.pipeline.Pipeline`, where they are bypassed automatically at
predict time -- the single robust way to avoid leaking synthetic samples into
evaluation.
"""
from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import config


def make_smote(k_neighbors: int = 5) -> SMOTE:
    """Return a SMOTE over-sampler that balances every class to the majority.

    ``k_neighbors`` controls how many minority neighbours each synthetic point
    is interpolated from; it must stay below the smallest class size.
    """
    return SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)


def make_undersampler() -> RandomUnderSampler:
    """Return a random under-sampler that trims every class to the smallest."""
    return RandomUnderSampler(random_state=config.RANDOM_STATE)


# Registry consumed by the imbalance experiment.  ``None`` is the untreated
# baseline used for comparison.
SAMPLERS: dict[str, object] = {
    "none": None,
    "SMOTE": make_smote(),
    "undersample": make_undersampler(),
}


def class_distribution(labels) -> pd.Series:
    """Return per-class counts ordered by :data:`config.CLASS_NAMES`.

    A tiny helper used by the EDA script and the imbalance report so the
    distribution is always presented in the same, readable order.
    """
    counts = pd.Series(labels).value_counts()
    return counts.reindex(config.CLASS_NAMES).fillna(0).astype(int)
