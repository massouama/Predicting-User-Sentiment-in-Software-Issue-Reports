"""Re-run the 20 vectorizer x classifier combinations and dump, for each,
the exact 3x3 confusion matrix counts and the per-class precision/recall/F1,
so the report can comment on every matrix with real numbers.

Output: scratchpad/per_combo_metrics.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src import models
from src.experiment import _CV, _MEMORY, prepare_data
from src.vectorization import VECTORIZERS, pretrained_available

OUT = Path(sys.argv[1])

X_train, X_test, y_train, y_test, df = prepare_data()
labels = list(config.CLASS_NAMES)

# Class counts in the test set (for context in commentary).
test_counts = {c: int((y_test == c).sum()) for c in labels}
train_counts = {c: int((y_train == c).sum()) for c in labels}

vec_names = ["BoW", "TF-IDF", "Word2Vec"]
if not os.environ.get("SKIP_PRETRAINED") and pretrained_available():
    vec_names.append("Word2Vec-pretrained")

results = {
    "test_counts": test_counts,
    "train_counts": train_counts,
    "n_train": int(len(y_train)),
    "n_test": int(len(y_test)),
    "combos": [],
}

for vec_name in vec_names:
    dense = VECTORIZERS[vec_name]["dense"]
    classifiers = models.build_classifiers(dense=dense)
    for clf_name, spec in classifiers.items():
        pipe = Pipeline(
            steps=[("vec", VECTORIZERS[vec_name]["factory"]()), (models.CLF_STEP, spec["estimator"])],
            memory=_MEMORY,
        )
        search = GridSearchCV(pipe, spec["param_grid"], scoring=config.SCORING, cv=_CV, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        p, r, f, s = precision_recall_fscore_support(y_test, y_pred, labels=labels, zero_division=0)
        acc = float((y_pred == y_test).mean())

        results["combos"].append({
            "vectorizer": vec_name,
            "classifier": clf_name,
            "cv_f1": float(search.best_score_),
            "accuracy": acc,
            "confusion_matrix": cm.tolist(),   # rows = true (neg,neu,pos), cols = predicted
            "per_class": {
                labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                            "f1": float(f[i]), "support": int(s[i])}
                for i in range(len(labels))
            },
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        })
        print(f"done: {vec_name} + {clf_name}")

OUT.write_text(json.dumps(results, indent=2))
print(f"\nWrote {OUT}")
