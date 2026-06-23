"""Add **BERT** to the model comparison — runnable wherever the model hub is reachable.

Why a separate script?
----------------------
In this sandbox the Hugging Face hub is blocked, so BERT weights cannot be
downloaded and the main pipeline skips BERT automatically. On any machine with
internet access (e.g. **Google Colab**) this script produces real BERT results
that slot into the exact same comparison.

Run it with::

    pip install transformers torch          # the only extra dependencies
    python scripts/run_bert_experiment.py

What it does (consistent with `main.py`):
1. Loads the same cleaned train/test split (`prepare_data`).
2. Encodes every document **once** with a frozen pre-trained BERT
   (mean-pooled embeddings) — a pre-trained model is external knowledge, so
   embedding the whole corpus up front introduces no label leakage.
3. Tunes the same five classifiers (dense variants) with the same 5-fold CV and
   `GridSearchCV`, evaluates on the held-out test set.
4. **Appends** the BERT rows to ``results/tables/model_comparison.csv`` (re-sorted
   by macro-F1) and saves a confusion matrix per model.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

import config
from src import models
from src.evaluation import compute_metrics, plot_confusion_matrix
from src.experiment import _CV, prepare_data
from src.vectorization import BertVectorizer, bert_available


def main(model_name: str = "distilbert-base-uncased") -> None:
    if not bert_available():
        print(
            "transformers / torch not installed. Install them first:\n"
            "    pip install transformers torch\n"
            "Then re-run this script (needs internet access to the model hub)."
        )
        return

    config.ensure_directories()
    X_train, X_test, y_train, y_test, _ = prepare_data()

    # Encode once (frozen model -> no leakage from embedding the whole corpus).
    print(f"Encoding {len(X_train)} train + {len(X_test)} test docs with {model_name} ...")
    bert = BertVectorizer(model_name=model_name).fit(X_train)
    emb_train, emb_test = bert.transform(X_train), bert.transform(X_test)

    rows = []
    for name, spec in models.build_classifiers(dense=True).items():
        # Wrap in a 1-step pipeline so the existing "clf__" grids apply unchanged.
        search = GridSearchCV(
            Pipeline([(models.CLF_STEP, spec["estimator"])]),
            spec["param_grid"], scoring=config.SCORING, cv=_CV, n_jobs=-1,
        ).fit(emb_train, y_train)

        y_pred = search.predict(emb_test)
        metrics = compute_metrics(y_test, y_pred)
        rows.append({"vectorizer": "BERT", "model": name,
                     "cv_f1_macro": search.best_score_, **metrics,
                     "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()}})
        plot_confusion_matrix(y_test, y_pred, f"BERT + {name}",
                              config.CONFUSION_DIR / f"BERT_{name}.png")
        print(f"  {name:<20} CV f1={search.best_score_:.3f}  test f1={metrics['f1_macro']:.3f}")

    # Merge BERT rows into the comparison table (drop any previous BERT rows first).
    table = config.TABLES_DIR / "model_comparison.csv"
    df = pd.read_csv(table) if table.exists() else pd.DataFrame()
    df = df[df.get("vectorizer") != "BERT"] if len(df) else df
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df = df.sort_values("f1_macro", ascending=False).reset_index(drop=True)
    df.to_csv(table, index=False)
    print(f"\nUpdated {table} with {len(rows)} BERT rows.")


if __name__ == "__main__":
    main()
