"""experiences.py — Orchestration de toutes les expériences.

Ce fichier fait le lien entre les briques (nettoyage, vectorisation,
modèles, évaluation) et produit tous les résultats demandés :

  1. Comparaison principale : vectoriseur × classifieurs, réglés par GridSearchCV
  2. Déséquilibre de classes : aucun vs SMOTE vs sous-échantillonnage
  3. Réduction de dimension : TruncatedSVD (PCA du texte / LSA)
  4. Élagage de l'arbre de décision
  5. Arrêt précoce (early stopping) du gradient boosting
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import config
from src.donnees import charger_avis_facebook, diviser_donnees
from src.evaluation import (
    calculer_metriques,
    tracer_barres,
    tracer_comparaison,
    tracer_ligne,
    tracer_matrice_confusion,
)
from src.modeles import construire_classifieurs, construire_early_stopping
from src.nettoyage import Preprocesseur
from src.vectorisation import VECTORISEURS, creer_tfidf

# ── Validation croisée partagée ───────────────────────────────────────────────
# Un seul objet CV réutilisé partout : mêmes plis pour tous les modèles,
# résultats comparables et reproductibles (graine fixe).
_CV = StratifiedKFold(
    n_splits=config.NB_PLIS,
    shuffle=True,
    random_state=config.GRAINE,
)


def preparer_donnees():
    """Charge, nettoie et divise les données une seule fois.

    Le nettoyage est fait ici (hors CV) parce qu'il est stateless :
    il n'apprend rien des données, donc l'appliquer une fois est équivalent
    à l'appliquer à chaque pli (et beaucoup plus rapide).

    Renvoie : X_train, X_test, y_train, y_test, df_complet
    """
    df = charger_avis_facebook()

    # Nettoyage du texte.
    prep = Preprocesseur()
    df["texte_propre"] = prep.transform(df["text"].tolist())

    # On supprime les textes qui deviennent vides après nettoyage
    # (avis en emoji, langue non-anglaise, etc.).
    df = df[df["texte_propre"].str.len() > 0].reset_index(drop=True)

    # Remplace le texte brut par sa version nettoyée, puis divise.
    # (Un simple rename "texte_propre"->"text" créerait une colonne "text"
    # en double, et df["text"] renverrait alors un tableau 2D qui casse les
    # vectoriseurs BoW/TF-IDF.)
    df["text"] = df["texte_propre"]
    df = df.drop(columns=["texte_propre"])
    X_train, X_test, y_train, y_test = diviser_donnees(df)
    return X_train, X_test, y_train, y_test, df


# ── Expérience 1 : comparaison principale ────────────────────────────────────
def _regler_et_evaluer(nom_vec, dense, X_train, X_test, y_train, y_test):
    """Règle et évalue tous les classifieurs pour un vectoriseur donné.

    Pour chaque classifieur :
      1. Construit un pipeline : vectoriseur → classifieur
      2. Lance GridSearchCV (5-fold, scoré sur macro-F1)
      3. Évalue la meilleure config sur le test final
      4. Sauvegarde la matrice de confusion

    Renvoie une liste de dicts (une ligne par classifieur).
    """
    lignes = []
    classifieurs = construire_classifieurs(dense=dense)
    for nom_clf, spec in classifieurs.items():
        # Pipeline : vectoriseur → classifieur.
        pipe = Pipeline([
            ("vec", VECTORISEURS[nom_vec]["fabrique"]()),
            ("clf", spec["estimateur"]),
        ])

        # GridSearchCV teste toutes les combinaisons d'hyperparamètres
        # en validation croisée et garde la meilleure selon le macro-F1.
        recherche = GridSearchCV(
            pipe, spec["grille"],
            scoring=config.METRIQUE,
            cv=_CV,
            n_jobs=-1,    # utilise tous les cœurs disponibles
            refit=True,   # réentraîne le meilleur modèle sur tout le train
        )
        recherche.fit(X_train, y_train)

        y_pred = recherche.predict(X_test)
        metriques = calculer_metriques(y_test, y_pred)

        lignes.append({
            "vectoriseur":  nom_vec,
            "modele":       nom_clf,
            "cv_f1_macro":  recherche.best_score_,   # score moyen sur les 5 plis
            **metriques,
            "meilleurs_params": {
                k.replace("clf__", ""): v
                for k, v in recherche.best_params_.items()
            },
        })

        tracer_matrice_confusion(
            y_test, y_pred,
            titre=f"{nom_vec} + {nom_clf}",
            chemin=config.MATRICES_DIR / f"{nom_vec}_{nom_clf}.png",
        )
        print(f"    {nom_clf:<22} CV f1={recherche.best_score_:.3f}  "
              f"test f1={metriques['f1_macro']:.3f}")
    return lignes


def comparaison_principale() -> pd.DataFrame:
    """Lance la comparaison vectoriseur × classifieur et sauvegarde les résultats."""
    X_train, X_test, y_train, y_test, _ = preparer_donnees()

    toutes_lignes = []
    for nom_vec in VECTORISEURS:
        print(f"\n[{nom_vec}]")
        dense = VECTORISEURS[nom_vec]["dense"]
        toutes_lignes += _regler_et_evaluer(
            nom_vec, dense, X_train, X_test, y_train, y_test
        )

    resultats = (
        pd.DataFrame(toutes_lignes)
        .sort_values("f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    resultats.to_csv(config.TABLES_DIR / "comparaison_modeles.csv", index=False)
    tracer_comparaison(resultats, "f1_macro",   config.FIGURES_DIR / "comparaison_f1.png")
    tracer_comparaison(resultats, "accuracy",   config.FIGURES_DIR / "comparaison_accuracy.png")
    return resultats


# ── Expérience 2 : déséquilibre de classes ───────────────────────────────────
def experience_desequilibre() -> pd.DataFrame:
    """Compare trois stratégies face au déséquilibre de classes.

    Classifieur fixe : Régression logistique (on ne veut changer qu'une variable).
    Les trois stratégies testées :
      - "aucun"          : rien, ligne de base
      - "SMOTE"          : création d'exemples neutres synthétiques (k=5 voisins)
      - "sous-echant."   : suppression d'exemples des classes majoritaires

    IMPORTANT : le sampler est dans le pipeline imblearn, donc il ne touche
    jamais les données de validation/test → pas de fuite de données.
    """
    X_train, X_test, y_train, y_test, _ = preparer_donnees()

    # Les trois stratégies : None = pas de rééchantillonnage.
    strategies = {
        "aucun":        None,
        "SMOTE":        SMOTE(random_state=config.GRAINE, k_neighbors=5),
        "sous-echant.": RandomUnderSampler(random_state=config.GRAINE),
    }

    lignes = []
    for nom, sampler in strategies.items():
        # Construit le pipeline avec ou sans sampler.
        etapes = [("vec", creer_tfidf())]
        if sampler is not None:
            etapes.append(("sampler", sampler))   # actif seulement au fit, pas au predict
        etapes.append(("clf", LogisticRegression(max_iter=1000, random_state=config.GRAINE)))

        pipe = ImbPipeline(etapes)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        # Rappel par classe : montre si la classe neutre est retrouvée.
        rappel_par_classe = recall_score(
            y_test, y_pred, average=None,
            labels=list(config.CLASSES), zero_division=0
        )
        lignes.append({
            "strategie": nom,
            **calculer_metriques(y_test, y_pred),
            **{f"rappel_{c}": r for c, r in zip(config.CLASSES, rappel_par_classe)},
        })

    resultat = pd.DataFrame(lignes)
    resultat.to_csv(config.TABLES_DIR / "desequilibre.csv", index=False)
    tracer_barres(
        resultat.set_index("strategie")["f1_macro"],
        "Gestion du déséquilibre : macro-F1 (TF-IDF + LogReg)", "macro-F1",
        config.FIGURES_DIR / "desequilibre_f1.png", ylim=(0, 1),
    )
    return resultat


# ── Expérience 3 : réduction de dimension (SVD / PCA) ────────────────────────
def experience_pca() -> pd.DataFrame:
    """Étudie l'impact de TruncatedSVD (PCA du texte) sur les performances.

    On utilise TruncatedSVD et non PCA classique car PCA densifie la matrice
    creuse TF-IDF (très coûteux en mémoire). TruncatedSVD fait la même chose
    sans centrer les données → c'est la PCA standard pour le texte (LSA).

    On compare plusieurs nombres de composantes et on trace la courbe de
    variance expliquée cumulée.
    """
    X_train, X_test, y_train, y_test, _ = preparer_donnees()

    # On vectorise d'abord avec TF-IDF.
    tfidf = creer_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Référence : sans réduction (features complètes).
    base = LogisticRegression(max_iter=1000, random_state=config.GRAINE).fit(Xtr, y_train)
    f1_base = calculer_metriques(y_test, base.predict(Xte))["f1_macro"]

    # Variance expliquée cumulée sur toutes les composantes.
    max_comp = min(config.SVD_COMPOSANTES, Xtr.shape[1] - 1)
    svd_complet = TruncatedSVD(n_components=max_comp, random_state=config.GRAINE).fit(Xtr)
    var_cumulee = np.cumsum(svd_complet.explained_variance_ratio_)
    tracer_ligne(
        np.arange(1, max_comp + 1), var_cumulee,
        "Variance expliquée cumulée (SVD / PCA)", "composantes", "variance cumulée",
        config.FIGURES_DIR / "pca_variance.png",
    )

    # Test avec différents nombres de composantes.
    lignes = []
    for k in [50, 100, 200, max_comp]:
        svd = TruncatedSVD(n_components=k, random_state=config.GRAINE)
        Ztr, Zte = svd.fit_transform(Xtr), svd.transform(Xte)
        clf = LogisticRegression(max_iter=1000, random_state=config.GRAINE).fit(Ztr, y_train)
        lignes.append({
            "composantes":        k,
            "variance_expliquee": float(svd.explained_variance_ratio_.sum()),
            "f1_macro":           calculer_metriques(y_test, clf.predict(Zte))["f1_macro"],
        })
    # Ajoute la référence sans réduction.
    lignes.append({"composantes": Xtr.shape[1], "variance_expliquee": 1.0, "f1_macro": f1_base})

    resultat = pd.DataFrame(lignes)
    resultat.to_csv(config.TABLES_DIR / "pca.csv", index=False)
    tracer_ligne(
        resultat["composantes"], resultat["f1_macro"],
        "Macro-F1 selon le nombre de composantes SVD", "composantes", "macro-F1",
        config.FIGURES_DIR / "pca_f1.png",
    )
    return resultat


# ── Expérience 4 : élagage de l'arbre de décision ────────────────────────────
def experience_elagage() -> pd.DataFrame:
    """Trace le chemin de post-élagage par coût-complexité (ccp_alpha).

    Plus ccp_alpha est grand, plus on coupe de branches.
    On observe le compromis : score train ↘, score test ↗ (jusqu'à un certain point).
    """
    X_train, X_test, y_train, y_test, _ = preparer_donnees()
    tfidf = creer_tfidf()
    Xtr = tfidf.fit_transform(X_train)
    Xte = tfidf.transform(X_test)

    # Calcule tous les seuils d'élagage possibles.
    arbre_base = DecisionTreeClassifier(random_state=config.GRAINE)
    alphas = arbre_base.cost_complexity_pruning_path(Xtr, y_train).ccp_alphas
    # On échantillonne ~15 valeurs sur le chemin (le dernier alpha donne un seul nœud).
    alphas = np.unique(alphas[:-1])
    if len(alphas) > 15:
        alphas = alphas[:: max(1, len(alphas) // 15)]

    lignes = []
    for alpha in alphas:
        arbre = DecisionTreeClassifier(
            random_state=config.GRAINE, ccp_alpha=alpha
        ).fit(Xtr, y_train)
        lignes.append({
            "ccp_alpha":      float(alpha),
            "nb_noeuds":      int(arbre.tree_.node_count),
            "score_train":    accuracy_score(y_train, arbre.predict(Xtr)),
            "score_test":     accuracy_score(y_test,  arbre.predict(Xte)),
        })

    resultat = pd.DataFrame(lignes)
    resultat.to_csv(config.TABLES_DIR / "elagage.csv", index=False)

    # Graphique train vs test selon alpha.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(resultat["ccp_alpha"], resultat["score_train"], marker="o", label="train")
    ax.plot(resultat["ccp_alpha"], resultat["score_test"],  marker="s", label="test")
    ax.set_xlabel("ccp_alpha (force de l'élagage)")
    ax.set_ylabel("accuracy")
    ax.set_title("Post-élagage de l'arbre de décision (TF-IDF)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "elagage.png", dpi=config.DPI_FIGURE)
    plt.close(fig)
    return resultat


# ── Expérience 5 : early stopping ────────────────────────────────────────────
def experience_early_stopping() -> dict:
    """Montre que le Gradient Boosting s'arrête avant le plafond de 500 arbres.

    On réduit d'abord les features TF-IDF avec SVD (100 composantes),
    puis on entraîne le GradientBoosting avec n_iter_no_change=10 :
    dès que le score de validation stagne 10 tours, l'entraînement s'arrête.
    """
    X_train, X_test, y_train, y_test, _ = preparer_donnees()

    # Réduction de dimension pour accélérer le boosting.
    tfidf = creer_tfidf()
    svd   = TruncatedSVD(n_components=100, random_state=config.GRAINE)
    Xtr = svd.fit_transform(tfidf.fit_transform(X_train))
    Xte = svd.transform(tfidf.transform(X_test))

    gb = construire_early_stopping().fit(Xtr, y_train)

    # Accuracy au fil des itérations (pour visualiser où l'arrêt se produit).
    scores_etapes = [accuracy_score(y_test, p) for p in gb.staged_predict(Xte)]
    tracer_ligne(
        np.arange(1, len(scores_etapes) + 1), scores_etapes,
        f"Gradient Boosting — arrêt à {gb.n_estimators_} arbres (plafond : 500)",
        "itérations", "accuracy test",
        config.FIGURES_DIR / "early_stopping.png",
    )

    resume = {
        "plafond_arbres":  gb.get_params()["n_estimators"],
        "arbres_utilises": int(gb.n_estimators_),
        "accuracy_test":   accuracy_score(y_test, gb.predict(Xte)),
        "f1_macro_test":   calculer_metriques(y_test, gb.predict(Xte))["f1_macro"],
    }
    pd.DataFrame([resume]).to_csv(config.TABLES_DIR / "early_stopping.csv", index=False)
    return resume
