"""Run the complete sentiment-classification study end to end.

Usage
-----
    python main.py            # full run (recommended)
    python main.py --fast     # skip the slowest extras (early stopping)

Stages
------
1. Generate / load the synthetic issue-report corpus.
2. Exploratory figures + vectoriser x classifier comparison (tuned, 5-fold CV).
3. Class-imbalance handling (SMOTE vs under-sampling).
4. Dimensionality reduction (TruncatedSVD / PCA).
5. Decision-tree post-pruning path.
6. Early stopping in gradient boosting.

Every artefact (tables, confusion matrices, figures) is written under
``results/``.  A concise summary is printed at the end so the run is easy to
sanity-check.
"""
from __future__ import annotations

import argparse
import time

import config
from src import experiment


def _stage(title: str) -> float:
    """Print a banner for a stage and return its start time."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    return time.perf_counter()


def main(fast: bool = False) -> None:
    config.ensure_directories()
    experiment.clear_cache()  # start from a clean cache for honest timing
    t0 = time.perf_counter()

    start = _stage("STAGE 1-2  Data, EDA and main model comparison (tuned, 5-fold CV)")
    results = experiment.run_main_comparison()
    print(f"\nTop result: {results.iloc[0]['vectorizer']} + {results.iloc[0]['model']} "
          f"(test macro-F1 = {results.iloc[0]['f1_macro']:.3f})")
    print(f"[stage time: {time.perf_counter() - start:.1f}s]")

    start = _stage("STAGE 3  Class-imbalance handling (none vs SMOTE vs under-sampling)")
    imb = experiment.experiment_imbalance()
    print(imb.to_string(index=False))
    print(f"[stage time: {time.perf_counter() - start:.1f}s]")

    start = _stage("STAGE 4  Dimensionality reduction (TruncatedSVD / PCA)")
    pca = experiment.experiment_pca()
    print(pca.to_string(index=False))
    print(f"[stage time: {time.perf_counter() - start:.1f}s]")

    start = _stage("STAGE 5  Decision-tree post-pruning path")
    pruning = experiment.experiment_pruning()
    print(pruning.to_string(index=False))
    print(f"[stage time: {time.perf_counter() - start:.1f}s]")

    if not fast:
        start = _stage("STAGE 6  Early stopping in gradient boosting")
        es = experiment.experiment_early_stopping()
        print(es)
        print(f"[stage time: {time.perf_counter() - start:.1f}s]")

    print("\n" + "=" * 78)
    print("MODEL COMPARISON (sorted by test macro-F1)")
    print("=" * 78)
    cols = ["vectorizer", "model", "cv_f1_macro", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    print(results[cols].to_string(index=False))
    print(f"\nTotal runtime: {time.perf_counter() - t0:.1f}s")
    print(f"Artefacts written under: {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the slowest extra experiment")
    main(**vars(parser.parse_args()))
