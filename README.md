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
python scripts/explore_data.py          # summary stats + EDA figures

# 3. Run the whole study end to end (~8 minutes on 4 cores)
python main.py                          # add --fast to skip the slowest extra
```

All outputs are written under [`results/`](results/): a model-comparison table,
confusion matrices, comparison charts and the ablation figures referenced in the
report.

```bash
# 4. (optional) Rebuild the PDF report from the Markdown source
python scripts/build_report_pdf.py     # -> report/REPORT.pdf
```

### Notes for a local run

- **NLTK data** (stop-words, WordNet) downloads automatically on first run; it is
  small and needs a working internet connection once.
- **Pre-trained Word2Vec.** The `Word2Vec-pretrained` vectoriser uses the
  Google-News vectors, which gensim downloads **once (~1.7 GB)**. To skip that
  download for a fast local run, set an environment variable:

  ```bash
  SKIP_PRETRAINED=1 python main.py      # runs BoW, TF-IDF and in-domain Word2Vec only
  ```

  With no internet access the pre-trained vectoriser is skipped automatically.
- The dataset is already bundled (`data/issue_sentiment.csv`), so `python main.py`
  runs offline apart from the optional pre-trained download above.

---

## 2. The dataset

By default the project uses a **real, pre-cleaned review dataset** — the *"App
Store reviews"* option of the brief: a Kaggle export of **Facebook app reviews**
(`content` + 1–5 `score`), bundled at
[`data/facebook_reviews.csv`](data/facebook_reviews.csv) and loaded by
[`src/real_data.py`](src/real_data.py). Scores map to sentiment (1–2★ → negative,
3★ → neutral, 4–5★ → positive). After de-duplicating the review text (so identical
strings can't leak across the split) **5 923 unique reviews** remain:

| Class | From | Share |
|-------|------|-------|
| `negative` | 1–2 ★ | ~31 % |
| `neutral`  | 3 ★ | ~5 % |
| `positive` | 4–5 ★ | ~64 % |

The strong, natural positive skew is what motivates the SMOTE / under-sampling
experiment, and the tiny 3★ "neutral" class genuinely overlaps its neighbours — a
real, non-trivial learning problem.

---

## 3. Pipeline overview

```
raw text
   │  src/preprocessing.py
   ▼  clean → tokenise → drop stop-words (keep negations) → WordNet lemmatise
cleaned text
   │  src/vectorization.py
   ▼  Bag-of-Words │ TF-IDF │ Word2Vec │ Word2Vec-pretrained (Google-News)
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
| Pre-cleaned dataset (Facebook app reviews) | `src/real_data.py` |
| Cleaning, lemmatisation, stop-words | `src/preprocessing.py` |
| BoW, TF-IDF, Word2Vec, **pre-trained Word2Vec** (Google-News) | `src/vectorization.py` |
| SMOTE / under-sampling | `src/balancing.py`, `experiment_imbalance` |
| ≥3 classifiers (NB, Decision Tree, ensembles) | `src/models.py` |
| 5-fold cross-validation + GridSearchCV tuning | `src/experiment.py` |
| Decision-tree pre- **and** post-pruning | `models.py` grid + `experiment_pruning` |
| PCA / dimensionality reduction | `experiment_pca` |
| Regularisation | `LogisticRegression` (`C` tuned) |
| Early stopping (boosting) | `experiment_early_stopping` |
| Accuracy / Precision / Recall / F1 / confusion matrix | `src/evaluation.py` |

---

## 5. Project layout

```
.
├── config.py               # single source of truth (paths, seed, grids sizes…)
├── main.py                 # runs the full study end to end
├── requirements.txt
├── data/
│   ├── facebook_reviews.csv # real Facebook reviews (default source)
│   └── issue_sentiment.csv  # cached processed corpus
├── src/
│   ├── real_data.py        # Facebook review loader (cached)
│   ├── preprocessing.py     # text cleaning / lemmatisation
│   ├── vectorization.py     # BoW / TF-IDF / Word2Vec / pre-trained Word2Vec
│   ├── balancing.py         # SMOTE / under-sampling
│   ├── models.py            # classifiers + hyper-parameter grids
│   ├── evaluation.py        # metrics + figures
│   └── experiment.py        # orchestration of every experiment
├── scripts/
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
