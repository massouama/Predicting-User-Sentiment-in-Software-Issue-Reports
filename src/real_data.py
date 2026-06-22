"""Loader for the **real** app-review sentiment corpus.

The dataset is the Google Play review collection released with the
``sealuzh/user_quality`` software-engineering research repository: 288 k reviews
across 395 apps, each with a 1-5 star rating.  It is downloaded once (no
credentials needed), then turned into a labelled ``text,label`` corpus by mapping
star ratings onto the three sentiment classes (:data:`config.STAR_TO_SENTIMENT`).

This directly satisfies the project brief's *"App Store reviews dataset"* option.
Because the raw file is ~40 MB we keep only a reproducible, stratified sample
(:data:`config.N_SAMPLES_REAL`) that preserves the natural -- strongly
positive-skewed -- class imbalance, which is exactly what the SMOTE /
under-sampling experiment is meant to tackle.
"""
from __future__ import annotations

import urllib.request

import pandas as pd

import config


def _download_raw() -> None:
    """Fetch the raw reviews CSV once, caching it under ``data/``."""
    if config.RAW_REVIEWS_PATH.exists():
        return
    config.ensure_directories()
    urllib.request.urlretrieve(config.REVIEWS_URL, config.RAW_REVIEWS_PATH)


def build_app_reviews_dataset(
    n_samples: int = config.N_SAMPLES_REAL,
    random_state: int = config.RANDOM_STATE,
) -> pd.DataFrame:
    """Build a labelled ``text,label`` sample from the raw app reviews.

    Steps: download (if needed) -> drop empty reviews -> map stars to sentiment
    -> drop duplicate texts -> take a class-stratified sample that keeps the
    natural imbalance.  Returns a shuffled DataFrame.
    """
    _download_raw()
    raw = pd.read_csv(config.RAW_REVIEWS_PATH, usecols=["review", "star"])

    # Map stars -> sentiment and keep only usable, non-empty, deduplicated text.
    raw["label"] = raw["star"].map(config.STAR_TO_SENTIMENT)
    raw = raw.dropna(subset=["review", "label"])
    raw["text"] = raw["review"].astype(str).str.strip()
    raw = raw[raw["text"].str.len() > 0].drop_duplicates(subset="text")

    df = raw[["text", "label"]]

    # Class-stratified down-sample that preserves the natural proportions.
    if n_samples and n_samples < len(df):
        frac = n_samples / len(df)
        df = df.groupby("label", group_keys=False).sample(frac=frac, random_state=random_state)

    return df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    data = build_app_reviews_dataset()
    print(f"Built {len(data)} labelled app reviews")
    print("\nClass distribution:")
    print(data["label"].value_counts().to_string())
    print("\nExamples:")
    for _, row in data.head(6).iterrows():
        print(f"  [{row['label']:>8}] {row['text'][:90]}")
