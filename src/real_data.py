"""Loaders for the **real** review-sentiment corpora.

Two ready-to-use real datasets are supported (selected via
:data:`config.DATASET_SOURCE`):

* ``"facebook"`` *(default)* -- a Kaggle export of **Facebook app reviews**
  (``content`` + 1-5 ``score``), shipped with the repo at
  :data:`config.FACEBOOK_CSV_PATH`.
* ``"app_reviews"`` -- the Google Play review collection from the open
  ``sealuzh/user_quality`` research repository (288 k reviews), downloaded once
  (credential-free).

Both are turned into a labelled ``text,label`` corpus by the shared
:func:`reviews_to_labelled` helper, which maps star ratings onto the three
sentiment classes (:data:`config.STAR_TO_SENTIMENT`) and keeps a reproducible,
class-stratified sample that preserves the natural -- strongly positive-skewed --
imbalance. That imbalance is exactly what the SMOTE / under-sampling experiment
addresses. This satisfies the project brief's *"App Store reviews dataset"*
option with genuinely real data.
"""
from __future__ import annotations

import urllib.request

import pandas as pd

import config


def reviews_to_labelled(
    df: pd.DataFrame,
    text_col: str,
    score_col: str,
    n_samples: int | None,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Convert a raw reviews frame into a labelled ``text,label`` sample.

    Maps the star ``score_col`` to sentiment, drops empty / duplicate texts, and
    optionally takes a class-stratified sample of ``n_samples`` that preserves
    the natural class proportions (``None`` keeps every row). Returns a shuffled
    DataFrame.
    """
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str).str.strip(),
            "label": df[score_col].map(config.STAR_TO_SENTIMENT),
        }
    )
    out = out.dropna(subset=["label"])
    out = out[out["text"].str.len() > 0].drop_duplicates(subset="text")

    if n_samples and n_samples < len(out):
        frac = n_samples / len(out)
        out = out.groupby("label", group_keys=False).sample(frac=frac, random_state=random_state)

    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def build_facebook_reviews_dataset(
    path=config.FACEBOOK_CSV_PATH,
    n_samples: int | None = config.FACEBOOK_N_SAMPLES,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build the labelled corpus from the bundled Facebook-reviews CSV."""
    raw = pd.read_csv(path, usecols=[config.FACEBOOK_TEXT_COL, config.FACEBOOK_SCORE_COL])
    return reviews_to_labelled(
        raw, config.FACEBOOK_TEXT_COL, config.FACEBOOK_SCORE_COL, n_samples, random_state
    )


def build_app_reviews_dataset(
    n_samples: int = config.N_SAMPLES_REAL,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build the labelled corpus from the Google Play (``sealuzh``) reviews."""
    if not config.RAW_REVIEWS_PATH.exists():
        config.ensure_directories()
        urllib.request.urlretrieve(config.REVIEWS_URL, config.RAW_REVIEWS_PATH)
    raw = pd.read_csv(config.RAW_REVIEWS_PATH, usecols=["review", "star"])
    return reviews_to_labelled(raw, "review", "star", n_samples, random_state)


if __name__ == "__main__":
    data = build_facebook_reviews_dataset()
    print(f"Built {len(data)} labelled Facebook reviews")
    print("\nClass distribution:")
    print(data["label"].value_counts().to_string())
    print("\nExamples:")
    for _, row in data.head(6).iterrows():
        print(f"  [{row['label']:>8}] {row['text'][:90]}")
