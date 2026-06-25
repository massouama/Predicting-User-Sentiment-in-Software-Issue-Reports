"""Orchestration de bout en bout de toutes les expériences du projet.

Transforme les briques (:mod:`preprocessing`, :mod:`vectorization`,
:mod:`models`, ...) en les expériences demandées par le sujet :

* :func:`run_main_comparison`       -- grille vectoriseur x classifieur (réglage + CV)
* :func:`experiment_imbalance`      -- aucun vs SMOTE vs sous-échantillonnage
* :func:`experiment_pca`            -- réduction de dimension (TruncatedSVD / LSA)
* :func:`experiment_pruning`        -- post-élagage cost-complexity d'un arbre
* :func:`experiment_early_stopping` -- early stopping du gradient boosting

Deux choix de performance gardent la suite rapide tout en restant
méthodologiquement corrects :

1. **Nettoyer une seule fois.** Le nettoyage est sans état (il n'apprend rien),
   donc l'appliquer une fois en amont équivaut exactement à le refaire dans
   chaque pli de CV, sans relemmatiser le corpus des milliers de fois.
2. **Mettre les vectoriseurs en cache.** Un :class:`joblib.Memory` partagé met
   en cache chaque vectoriseur ajusté. Comme toutes les recherches réutilisent
   un même :class:`~sklearn.model_selection.StratifiedKFold`, les ajustements
   coûteux (Word2Vec / TF-IDF) sont calculés une fois par pli puis réutilisés.
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
from src.real_data import load_or_create_dataset
from src.evaluation import (
    compute_metrics,
    plot_bar,
    plot_confusion_matrix,
    plot_line,
    plot_model_comparison,
)
from src.preprocessing import TextPreprocessor
from src.vectorization import VECTORIZERS, make_tfidf, pretrained_available

# Découpage CV fixe et partagé -> plis identiques pour toutes les recherches,
# ce qui rend efficace le cache joblib des vectoriseurs.
_CV = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

# Cache disque des transformateurs (vectoriseurs) ajustés.
_CACHE_DIR = config.RESULTS_DIR / ".joblib_cache"
_MEMORY = Memory(location=str(_CACHE_DIR), verbose=0)


def clear_cache() -> None:
    """Supprime le cache joblib des vectoriseurs (pour une exécution propre)."""
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR)


# --- Préparation des données -----------------------------------------------
def prepare_data():
    """Charge, nettoie et découpe le corpus une fois pour toutes les expériences.

    Renvoie ``X_train, X_test`` (documents nettoyés), ``y_train, y_test``
    (labels) et ``df`` (corpus complet avec une colonne ``clean``, pour l'EDA).
    """
    df = load_or_create_dataset()
    df["clean"] = TextPreprocessor().transform(df["text"].tolist())

    # Certains avis deviennent vides après nettoyage (non anglais, que des mots
    # vides, emoji seuls) : on les retire pour que chaque vectoriseur reçoive un
    # document non dégénéré.
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].to_numpy(),
        df["label"].to_numpy(),
        test_size=config.TEST_SIZE,
        stratify=df["label"],          # conserve les proportions de classes
        random_state=config.RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, df


# --- Analyse exploratoire (figures du rapport) -----------------------------
def run_eda(df: pd.DataFrame) -> None:
    """Sauvegarde les figures de distribution des classes et de longueur."""
    dist = class_distribution(df["label"])
    plot_bar(
        dist, "Distribution des classes (corpus complet)", "nombre d'avis",
        config.FIGURES_DIR / "class_distribution.png",
    )

    lengths = df["clean"].str.split().map(len)
    plot_bar(
        lengths.groupby(df["label"]).mean().reindex(config.CLASS_NAMES),
        "Nombre moyen de tokens nettoyés par classe", "tokens moyens",
        config.FIGURES_DIR / "token_length_by_class.png",
    )


# --- Comparaison principale : vectoriseur x classifieur, réglés en CV 5 plis ---
def _tune_and_evaluate(vec_name, dense, X_train, X_test, y_train, y_test, save_models):
    """Règle et teste chaque classifieur pour un vectoriseur. Renvoie les lignes."""
    rows = []
    classifiers = models.build_classifiers(dense=dense)
    for clf_name, spec in classifiers.items():
        pipe = Pipeline(
            steps=[("vec", VECTORIZERS[vec_name]["factory"]()), (models.CLF_STEP, spec["estimator"])],
            memory=_MEMORY,  # met en cache l'ajustement du vectoriseur entre plis / combos
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
    """Lance la comparaison vectoriseur x classifieur complète et écrit les artefacts.

    Pour chaque vectoriseur (BoW, TF-IDF, Word2Vec, et Word2Vec pré-entraîné si
    les vecteurs sont chargeables), chaque classifieur est réglé en CV à 5 plis
    sur le train puis évalué une fois sur le test. Matrices de confusion, table
    de résultats et diagrammes de comparaison sont écrits dans ``results/``.
    """
    X_train, X_test, y_train, y_test, df = prepare_data()
    run_eda(df)

    # Le Word2Vec pré-entraîné (Google News, ~1,7 Go via gensim-data) n'est ajouté
    # que si ses vecteurs sont chargeables. SKIP_PRETRAINED=1 saute ce
    # téléchargement pour une exécution locale rapide.
    vec_names = ["BoW", "TF-IDF", "Word2Vec"]
    if not os.environ.get("SKIP_PRETRAINED") and pretrained_available():
        vec_names.append("Word2Vec-pretrained")

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


# --- Expérience : gestion du déséquilibre de classes -----------------------
def experiment_imbalance() -> pd.DataFrame:
    """Compare aucun rééchantillonnage vs SMOTE vs sous-échantillonnage (TF-IDF).

    Utilise une régression logistique fixe comme base rapide, pour que seule la
    stratégie de rééchantillonnage change. Rapporte les métriques globales et le
    rappel par classe, là où les gains sur la classe minoritaire apparaissent.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    rows = []
    for name, sampler in SAMPLERS.items():
        steps = [("vec", make_tfidf())]
        if sampler is not None:
            steps.append(("sampler", sampler))           # actif uniquement à l'entraînement
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
        "Gestion du déséquilibre : macro-F1 (TF-IDF + LogReg)", "macro-F1",
        config.FIGURES_DIR / "imbalance_f1.png", ylim=(0, 1),
    )
    return result


# --- Expérience : réduction de dimension (PCA / TruncatedSVD) ---------------
def experiment_pca() -> pd.DataFrame:
    """Étudie TruncatedSVD (LSA) sur les features TF-IDF.

    La PCA ordinaire centre les données et densifie donc une matrice TF-IDF
    creuse, ce qui est coûteux en mémoire. :class:`TruncatedSVD` est la réduction
    équivalente standard pour le texte creux (elle ne centre pas) ; on l'appelle
    « PCA » par rapport au sujet.

    Produit la courbe de variance expliquée cumulée et le macro-F1 aval en
    fonction du nombre de composantes, comparés à la baseline pleine dimension.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()

    tfidf = make_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Baseline pleine dimension (sans réduction).
    base = LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE).fit(Xtr, y_train)
    baseline_f1 = compute_metrics(y_test, base.predict(Xte))["f1_macro"]

    max_components = min(config.SVD_COMPONENTS, Xtr.shape[1] - 1)
    svd_full = TruncatedSVD(n_components=max_components, random_state=config.RANDOM_STATE).fit(Xtr)
    cum_var = np.cumsum(svd_full.explained_variance_ratio_)
    plot_line(
        np.arange(1, max_components + 1), cum_var,
        "Variance expliquée cumulée (TruncatedSVD)", "composantes", "variance cumulée",
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
        "Macro-F1 vs composantes SVD (dernier point = TF-IDF complet)", "composantes", "macro-F1",
        config.FIGURES_DIR / "pca_f1.png",
    )
    return result


# --- Expérience : post-élagage de l'arbre (cost-complexity) -----------------
def experiment_pruning() -> pd.DataFrame:
    """Trace le post-élagage cost-complexity (``ccp_alpha``) d'un arbre.

    Un ``ccp_alpha`` plus grand élague plus agressivement et réduit l'arbre. On
    suit l'accuracy train/test et la taille de l'arbre le long du chemin
    d'élagage pour exposer le compromis surapprentissage / généralisation.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    tfidf = make_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    base_tree = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    alphas = base_tree.cost_complexity_pruning_path(Xtr, y_train).ccp_alphas
    # Échantillonne ~15 alphas le long du chemin (retire le dernier, arbre réduit à la racine).
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

    # Accuracy train vs test le long du chemin d'élagage.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["ccp_alpha"], result["train_accuracy"], marker="o", label="train")
    ax.plot(result["ccp_alpha"], result["test_accuracy"], marker="s", label="test")
    ax.set_xlabel("ccp_alpha (force d'élagage)")
    ax.set_ylabel("accuracy")
    ax.set_title("Post-élagage de l'arbre de décision (TF-IDF)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "pruning_accuracy.png", dpi=config.FIG_DPI)
    plt.close(fig)
    return result


# --- Expérience : early stopping du gradient boosting ----------------------
def experiment_early_stopping() -> dict:
    """Montre l'early stopping arrêtant le boosting avant son plafond de 500 arbres.

    Le TF-IDF est d'abord réduit par TruncatedSVD (features denses et compactes
    que le boosting gère efficacement). ``n_iter_no_change`` arrête alors
    l'entraînement dès que la performance de validation plafonne. On trace aussi
    l'accuracy test par étape pour visualiser où les arbres supplémentaires
    cessent d'aider.
    """
    X_train, X_test, y_train, y_test, _ = prepare_data()
    tfidf = make_tfidf()
    svd = TruncatedSVD(n_components=100, random_state=config.RANDOM_STATE)
    Xtr = svd.fit_transform(tfidf.fit_transform(X_train))
    Xte = svd.transform(tfidf.transform(X_test))

    gb = models.build_early_stopping_model().fit(Xtr, y_train)

    # Accuracy test par étape : un point par itération de boosting réellement entraînée.
    staged = [accuracy_score(y_test, p) for p in gb.staged_predict(Xte)]
    plot_line(
        np.arange(1, len(staged) + 1), staged,
        f"Accuracy test du gradient boosting (arrêté à {gb.n_estimators_} arbres)",
        "itérations de boosting", "accuracy test",
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
