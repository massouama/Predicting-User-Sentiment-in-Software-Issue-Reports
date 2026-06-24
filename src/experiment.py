"""End-to-end orchestration of every experiment in the project.

This module turns the building blocks (:mod:`preprocessing`,
:mod:`vectorization`, :mod:`models`, ...) into the concrete experiments the brief
asks for:

* :func:`run_main_comparison`     -- vectoriser x classifier grid with tuning + CV
* :func:`experiment_imbalance`    -- none vs SMOTE vs under-sampling
* :func:`experiment_pca`          -- dimensionality reduction (TruncatedSVD/LSA)
* :func:`experiment_pruning`      -- decision-tree cost-complexity post-pruning
* :func:`experiment_early_stopping` -- early stopping in gradient boosting

Two performance choices keep the whole suite fast while remaining methodologically
sound:

1. **Preprocess once.**  Text cleaning is *stateless* (it learns nothing from the
   data), so applying it a single time up front is exactly equivalent to running
   it inside every CV fold -- but avoids re-lemmatising the corpus thousands of
   times.
2. **Cache vectorisers.**  A shared :class:`joblib.Memory` caches each fitted
   vectoriser.  Because all searches reuse one fixed
   :class:`~sklearn.model_selection.StratifiedKFold`, the (expensive) Word2Vec /
   TF-IDF fits are computed once per fold and reused across every hyper-parameter
   combination and every classifier.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

import config
from src import models
from src.balancing import SAMPLERS, class_distribution
from src.data_generation import load_or_create_dataset
from src.evaluation import (
    compute_metrics,
    plot_bar,
    plot_confusion_matrix,
    plot_line,
    plot_model_comparison,
)
from src.preprocessing import TextPreprocessor
from src.vectorization import VECTORIZERS, bert_available, make_tfidf, pretrained_available

# Shared, fixed CV splitter -> identical folds across every search, which is
# what makes the joblib vectoriser cache effective.
_CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

# On-disk cache for fitted pipeline transformers (vectorisers).
_CACHE_DIR = config.RESULTS_DIR / ".joblib_cache"
_MEMORY = Memory(location=str(_CACHE_DIR), verbose=0)


def clear_cache() -> None:
    """Remove the joblib transformer cache (call for a clean, timed run)."""
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def prepare_data():
    """Load, clean and split the corpus once for every experiment.

    Returns
    -------
    X_train, X_test : list[str]
        Pre-cleaned documents (ready for any vectoriser).
    y_train, y_test : numpy.ndarray
        String labels.
    df : pandas.DataFrame
        The full dataset with an added ``clean`` column (used by the EDA step).
    """
    df = load_or_create_dataset()
    df["clean"] = TextPreprocessor().transform(df["text"].tolist())

    # Real reviews can reduce to an empty string (non-English, all stop-words,
    # emoji-only); drop those so every vectoriser sees a non-degenerate document.
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].to_numpy(),
        df["label"].to_numpy(),
        test_size=config.TEST_SIZE,
        stratify=df["label"],          # preserve class proportions in both splits
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, df


# ---------------------------------------------------------------------------
# Exploratory data analysis (figures for the report)
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Save class-distribution and document-length figures."""
    dist = class_distribution(df["label"])
    plot_bar(
        dist, "Class distribution (full corpus)", "number of reports",
        config.FIGURES_DIR / "class_distribution.png",
    )

    lengths = df["clean"].str.split().map(len)
    plot_bar(
        lengths.groupby(df["label"]).mean().reindex(config.CLASS_NAMES),
        "Mean cleaned-token count per class", "mean tokens",
        config.FIGURES_DIR / "token_length_by_class.png",
    )


# ---------------------------------------------------------------------------
# Main comparison: vectoriser x classifier, tuned with 5-fold CV
# ---------------------------------------------------------------------------
def _tune_and_evaluate(vec_name, dense, X_train, X_test, y_train, y_test, save_models):
    """Tune and test every classifier for one vectoriser. Returns result rows."""
    rows = []
    classifiers = models.build_classifiers(dense=dense)
    for clf_name, spec in classifiers.items():
        pipe = Pipeline(
            steps=[("vec", VECTORIZERS[vec_name]["factory"]()), (models.CLF_STEP, spec["estimator"])],
            memory=_MEMORY,  # cache the vectoriser fit across folds / param combos
        )
        search = GridSearchCV(
            pipe, spec["param_grid"], scoring=config.SCORING,
            cv=_CV, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        rows.append(
            {
                "vectorizer": vec_name,
                "model": clf_name,
                "cv_f1_macro": search.best_score_,
                **metrics,
                "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            }
        )
        plot_confusion_matrix(
            y_test, y_pred, f"{vec_name} + {clf_name}",
            config.CONFUSION_DIR / f"{vec_name}_{clf_name}.png",
        )
        if save_models:
            from joblib import dump

            dump(search.best_estimator_, config.MODELS_DIR / f"{vec_name}_{clf_name}.joblib")
        print(f"    {clf_name:<20} CV f1={search.best_score_:.3f}  test f1={metrics['f1_macro']:.3f}")
    return rows


def run_main_comparison(save_models: bool = False) -> pd.DataFrame:
    """Run the full vectoriser x classifier comparison and persist artefacts.

    For each vectoriser (BoW, TF-IDF, Word2Vec, and BERT when available) every
    classifier is hyper-parameter-tuned with 5-fold CV on the training split and
    evaluated once on the held-out test split.  Confusion matrices, a results
    table and comparison bar charts are written to ``results/``.
    """
    X_train, X_test, y_train, y_test, df = prepare_data()
    run_eda(df)

    # Pre-trained Word2Vec is included if the vectors can be loaded (Google-News,
    # downloaded once via gensim-data, ~1.7 GB). Set SKIP_PRETRAINED=1 to skip that
    # download for a quick local run. BERT is added only if its optional
    # deep-learning stack is importable.
    vec_names = ["BoW", "TF-IDF", "Word2Vec"]
    if not os.environ.get("SKIP_PRETRAINED") and pretrained_available():
        vec_names.append("Word2Vec-pretrained")
    if bert_available():
        vec_names.append("BERT")

    all_rows = []
    for vec_name in vec_names:
        print(f"\n[{vec_name}]")
        dense = VECTORIZERS[vec_name]["dense"]
        all_rows += _tune_and_evaluate(vec_name, dense, X_train, X_test, y_train, y_test, save_models)

    results = pd.DataFrame(all_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    results.to_csv(config.TABLES_DIR / "model_comparison.csv", index=False)
    plot_model_comparison(results, "f1_macro", config.FIGURES_DIR / "comparison_f1.png")
    plot_model_comparison(results, "accuracy", config.FIGURES_DIR / "comparison_accuracy.png")
    return results


# ---------------------------------------------------------------------------
# Experiment: class-imbalance handling
# ---------------------------------------------------------------------------
def experiment_imbalance() -> pd.DataFrame:
    """Compare no-resampling vs SMOTE vs under-sampling on TF-IDF features.

    Uses Logistic Regression as a fixed, fast base learner so the only thing
    that changes is the sampling strategy.  Reports overall metrics plus
    per-class recall, which is where minority-class gains show up.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    rows = []
    for name, sampler in SAMPLERS.items():
        steps = [("vec", make_tfidf())]
        if sampler is not None:
            steps.append(("sampler", sampler))           # active only at fit time
        steps.append((models.CLF_STEP, LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)))
        pipe = ImbPipeline(steps)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        per_class = recall_score(y_test, y_pred, average=None, labels=list(config.CLASS_NAMES), zero_division=0)
        rows.append(
            {
                "strategy": name,
                **compute_metrics(y_test, y_pred),
                **{f"recall_{cls}": r for cls, r in zip(config.CLASS_NAMES, per_class)},
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "imbalance_comparison.csv", index=False)
    plot_bar(
        result.set_index("strategy")["f1_macro"],
        "Imbalance handling: macro-F1 (TF-IDF + LogReg)", "macro-F1",
        config.FIGURES_DIR / "imbalance_f1.png", ylim=(0, 1),
    )
    return result


# ---------------------------------------------------------------------------
# Experiment: dimensionality reduction (PCA / TruncatedSVD)
# ---------------------------------------------------------------------------
def experiment_pca() -> pd.DataFrame:
    """Study TruncatedSVD (LSA) on TF-IDF features.

    Plain PCA centres the data and so densifies a sparse TF-IDF matrix, which is
    wasteful and memory-hungry.  :class:`TruncatedSVD` is the standard,
    PCA-equivalent reduction for sparse text (it skips centring), so we use it
    here and refer to it as PCA for the brief.

    Produces the cumulative explained-variance curve and downstream macro-F1 as
    a function of the number of components, compared against the full-feature
    baseline.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()

    tfidf = make_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Full-dimensional baseline (no reduction).
    base = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(Xtr, y_train)
    baseline_f1 = compute_metrics(y_test, base.predict(Xte))["f1_macro"]

    max_components = min(config.SVD_COMPONENTS, Xtr.shape[1] - 1)
    svd_full = TruncatedSVD(n_components=max_components, random_state=config.RANDOM_STATE).fit(Xtr)
    cum_var = np.cumsum(svd_full.explained_variance_ratio_)
    plot_line(
        np.arange(1, max_components + 1), cum_var,
        "TruncatedSVD cumulative explained variance", "components", "cumulative variance ratio",
        config.FIGURES_DIR / "pca_explained_variance.png",
    )

    rows = []
    for k in [50, 100, 200, max_components]:
        svd = TruncatedSVD(n_components=k, random_state=config.RANDOM_STATE)
        Ztr, Zte = svd.fit_transform(Xtr), svd.transform(Xte)
        clf = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(Ztr, y_train)
        rows.append(
            {
                "components": k,
                "explained_variance": float(svd.explained_variance_ratio_.sum()),
                "f1_macro": compute_metrics(y_test, clf.predict(Zte))["f1_macro"],
            }
        )
    rows.append({"components": Xtr.shape[1], "explained_variance": 1.0, "f1_macro": baseline_f1})

    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "pca_comparison.csv", index=False)
    plot_line(
        result["components"], result["f1_macro"],
        "Macro-F1 vs SVD components (last point = full TF-IDF)", "components", "macro-F1",
        config.FIGURES_DIR / "pca_f1.png",
    )
    return result


# ---------------------------------------------------------------------------
# Experiment: decision-tree post-pruning (cost-complexity)
# ---------------------------------------------------------------------------
def experiment_pruning() -> pd.DataFrame:
    """Trace cost-complexity (``ccp_alpha``) post-pruning of a decision tree.

    Larger ``ccp_alpha`` prunes more aggressively, shrinking the tree.  We track
    train/test accuracy and tree size along the pruning path to expose the
    classic over-fitting-vs-generalisation trade-off.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    tfidf = make_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    base_tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    alphas = base_tree.cost_complexity_pruning_path(Xtr, y_train).ccp_alphas
    # Sample up to ~15 alphas across the path (drop the final, fully-collapsed one).
    alphas = np.unique(alphas[:-1])
    if len(alphas) > 15:
        alphas = alphas[:: max(1, len(alphas) // 15)]

    rows = []
    for alpha in alphas:
        tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE, ccp_alpha=alpha).fit(Xtr, y_train)
        rows.append(
            {
                "ccp_alpha": float(alpha),
                "n_nodes": int(tree.tree_.node_count),
                "train_accuracy": accuracy_score(y_train, tree.predict(Xtr)),
                "test_accuracy": accuracy_score(y_test, tree.predict(Xte)),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(config.TABLES_DIR / "pruning_path.csv", index=False)

    # Train vs test accuracy along the pruning path.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["ccp_alpha"], result["train_accuracy"], marker="o", label="train")
    ax.plot(result["ccp_alpha"], result["test_accuracy"], marker="s", label="test")
    ax.set_xlabel("ccp_alpha (pruning strength)")
    ax.set_ylabel("accuracy")
    ax.set_title("Decision-tree post-pruning (TF-IDF)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pruning_accuracy.png", dpi=config.FIG_DPI)
    plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Experiment: early stopping in gradient boosting
# ---------------------------------------------------------------------------
def experiment_early_stopping() -> dict:
    """Show early stopping halting gradient boosting before its 500-tree ceiling.

    TF-IDF is first reduced with TruncatedSVD (dense, compact features that
    gradient boosting handles efficiently).  ``n_iter_no_change`` then stops
    training once validation performance plateaus.  We also plot the staged test
    accuracy to visualise where additional trees stop helping.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    tfidf = make_tfidf()
    svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_STATE)
    Xtr = svd.fit_transform(tfidf.fit_transform(X_train))
    Xte = svd.transform(tfidf.transform(X_test))

    gb = models.build_early_stopping_model().fit(Xtr, y_train)

    # Staged test accuracy: one point per boosting iteration actually trained.
    staged = [accuracy_score(y_test, p) for p in gb.staged_predict(Xte)]
    plot_line(
        np.arange(1, len(staged) + 1), staged,
        f"Gradient boosting test accuracy (early-stopped at {gb.n_estimators_} trees)",
        "boosting iterations", "test accuracy",
        config.FIGURES_DIR / "early_stopping.png",
    )

    summary = {
        "max_estimators": gb.get_params()["n_estimators"],
        "trees_used": int(gb.n_estimators_),
        "test_accuracy": accuracy_score(y_test, gb.predict(Xte)),
        "test_f1_macro": compute_metrics(y_test, gb.predict(Xte))["f1_macro"],
    }
    pd.DataFrame([summary]).to_csv(config.TABLES_DIR / "early_stopping.csv", index=False)
    return summary
