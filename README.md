# Predicting User Sentiment in Software Issue Reports

A complete, reproducible machine-learning pipeline that classifies the sentiment
of user-reported software issues (bug reports, feature requests, app reviews…)
into **negative**, **neutral** and **positive**.

The project compares several **text-vectorisation** strategies against several
**classifiers**, with full hyper-parameter tuning, cross-validation, class-
imbalance handling, dimensionality reduction, tree pruning and early stopping —
the complete checklist from the *ST2MLE — Machine Learning for IT Engineers*
brief.

---

## 1. Quick start

```bash
# 1. Install dependencies (Python 3.11 recommended)
pip install -r requirements.txt

# 2. (optional) Inspect the dataset
python scripts/generate_dataset.py     # (re)generate the cached corpus
python scripts/explore_data.py          # summary stats + EDA figures

# 3. Run the whole study end to end (~2 minutes on 4 cores)
python main.py                          # add --fast to skip the slowest extra
```

All outputs are written under [`results/`](results/): a model-comparison table,
confusion matrices, comparison charts and the ablation figures referenced in the
report.

```bash
# 4. (optional) Rebuild the PDF report from the Markdown source
python scripts/build_report_pdf.py     # -> report/REPORT.pdf
```

---

## 2. The dataset

Public sentiment-labelled issue corpora require API keys or downloads from hosts
that are unreachable in a sandboxed environment, so the project ships with a
**reproducible synthetic generator** ([`src/data_generation.py`](src/data_generation.py))
that mirrors the structure of real issue trackers:

| Class | Typical content | Share |
|-------|-----------------|-------|
| `negative` | crashes, bugs, frustration | ~53 % |
| `neutral`  | questions, feature requests, factual notes | ~30 % |
| `positive` | praise, thanks, "works great now" | ~17 % |

Three properties make it a *genuine* learning problem rather than trivial keyword
matching:

1. **Shared software vocabulary** across all classes (same components, versions,
   actions) — the model must learn *sentiment*, not topic.
2. **Ambiguity & mixed signals** — a fraction of reports use tone-neutral phrasing
   or borrow an opener/closing remark from another class (`AMBIGUITY_PROB`,
   `MIXING_PROB` in [`config.py`](config.py)).
3. **Label noise** (`LABEL_NOISE`) — a small fraction of labels are flipped,
   capping achievable accuracy below 100 %, exactly as human annotation would.

The deliberate class imbalance is what motivates the SMOTE / under-sampling
experiment. Everything is driven by a single seed (`RANDOM_STATE`), so the corpus
is byte-for-byte reproducible.

---

## 3. Pipeline overview

```
raw text
   │  src/preprocessing.py
   ▼  clean → tokenise → drop stop-words (keep negations) → WordNet lemmatise
cleaned text
   │  src/vectorization.py
   ▼  Bag-of-Words │ TF-IDF │ Word2Vec │ BERT (optional)
feature matrix
   │  src/balancing.py        (SMOTE / under-sampling, train folds only)
   │  TruncatedSVD            (PCA-equivalent reduction, optional)
   ▼  src/models.py
classifier  →  NaiveBayes │ DecisionTree │ RandomForest │ AdaBoost │ LogisticRegression
   │  GridSearchCV (5-fold, tuned)
   ▼  src/evaluation.py
metrics + confusion matrices + figures
```

### Design decisions worth highlighting

- **Preprocess once, then cache vectorisers.** Cleaning is stateless, so it runs
  a single time; a shared `joblib.Memory` plus one fixed `StratifiedKFold` means
  the expensive Word2Vec / TF-IDF fits are computed once per fold and reused
  across every hyper-parameter combination and classifier. Fast *and* leak-free.
- **Negation-aware stop-words.** `not`, `no`, `never`… are *kept*, because they
  flip sentiment.
- **Naive Bayes adapts to the feature space.** `MultinomialNB` for sparse,
  non-negative BoW/TF-IDF; `GaussianNB` for dense embeddings.
- **TruncatedSVD instead of PCA** for the sparse text matrices: it is the
  PCA-equivalent that avoids densifying a large sparse matrix (a.k.a. LSA).
- **Resampling lives inside an `imblearn` pipeline**, so it only ever touches the
  training folds — never the validation/test data.

---

## 4. What each requirement maps to

| Brief requirement | Where |
|---|---|
| Cleaning, lemmatisation, stop-words | `src/preprocessing.py` |
| BoW, TF-IDF, Word2Vec, BERT | `src/vectorization.py` |
| SMOTE / under-sampling | `src/balancing.py`, `experiment_imbalance` |
| ≥3 classifiers (NB, Decision Tree, ensembles) | `src/models.py` |
| 5-fold cross-validation + GridSearchCV tuning | `src/experiment.py` |
| Decision-tree pre- **and** post-pruning | `models.py` grid + `experiment_pruning` |
| PCA / dimensionality reduction | `experiment_pca` |
| Regularisation | `LogisticRegression` (`C` tuned) |
| Early stopping (boosting) | `experiment_early_stopping` |
| Accuracy / Precision / Recall / F1 / confusion matrix | `src/evaluation.py` |

> **BERT note.** A complete, documented `BertVectorizer` is provided. It needs the
> optional `transformers` + `torch` packages *and* network access to the model
> hub; where those are unavailable the experiments fall back to BoW / TF-IDF /
> Word2Vec and BERT is skipped automatically.

---

## 5. Project layout

```
.
├── config.py               # single source of truth (paths, seed, grids sizes…)
├── main.py                 # runs the full study end to end
├── requirements.txt
├── data/
│   └── issue_sentiment.csv # cached, reproducible corpus
├── src/
│   ├── data_generation.py  # synthetic issue-report generator
│   ├── preprocessing.py     # text cleaning / lemmatisation
│   ├── vectorization.py     # BoW / TF-IDF / Word2Vec / BERT
│   ├── balancing.py         # SMOTE / under-sampling
│   ├── models.py            # classifiers + hyper-parameter grids
│   ├── evaluation.py        # metrics + figures
│   └── experiment.py        # orchestration of every experiment
├── scripts/
│   ├── generate_dataset.py
│   ├── explore_data.py
│   └── build_report_pdf.py  # Markdown report -> PDF
├── results/                # tables, confusion matrices, figures (generated)
└── report/
    ├── REPORT.md            # full written report (source)
    └── REPORT.pdf           # rendered PDF (submission deliverable)
```

See [`report/REPORT.md`](report/REPORT.md) (or the rendered
[`report/REPORT.pdf`](report/REPORT.pdf)) for the dataset description, full
results tables, comparison discussion, trade-offs and future work.
