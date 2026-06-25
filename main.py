"""Lance l'étude complète de classification de sentiment de bout en bout.

Utilisation
-----------
    python main.py            # exécution complète (recommandé)
    python main.py --fast     # saute l'expérience la plus lente (early stopping)

Étapes
------
1. Chargement du corpus d'avis Facebook (mis en cache).
2. Figures exploratoires + comparaison vectoriseur x classifieur (réglée, CV 5 plis).
3. Gestion du déséquilibre (SMOTE vs sous-échantillonnage).
4. Réduction de dimension (TruncatedSVD / PCA).
5. Post-élagage de l'arbre de décision.
6. Early stopping du gradient boosting.

Tous les artefacts (tables, matrices de confusion, figures) sont écrits dans
``results/``. Un résumé concis est affiché à la fin.
"""
from __future__ import annotations

import argparse
import time

import config
from src import experiment


def _stage(title: str) -> float:
    """Affiche un bandeau d'étape et renvoie l'heure de début."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    return time.perf_counter()


def main(fast: bool = False) -> None:
    config.ensure_directories()
    experiment.clear_cache()  # part d'un cache propre pour un timing honnête
    t0 = time.perf_counter()

    start = _stage("ÉTAPE 1-2  Données, EDA et comparaison principale (réglée, CV 5 plis)")
    results = experiment.run_main_comparison()
    print(f"\nMeilleur résultat : {results.iloc[0]['vectorizer']} + {results.iloc[0]['model']} "
          f"(test macro-F1 = {results.iloc[0]['f1_macro']:.3f})")
    print(f"[durée de l'étape : {time.perf_counter() - start:.1f}s]")

    start = _stage("ÉTAPE 3  Gestion du déséquilibre (aucun vs SMOTE vs sous-échantillonnage)")
    imb = experiment.experiment_imbalance()
    print(imb.to_string(index=False))
    print(f"[durée de l'étape : {time.perf_counter() - start:.1f}s]")

    start = _stage("ÉTAPE 4  Réduction de dimension (TruncatedSVD / PCA)")
    pca = experiment.experiment_pca()
    print(pca.to_string(index=False))
    print(f"[durée de l'étape : {time.perf_counter() - start:.1f}s]")

    start = _stage("ÉTAPE 5  Post-élagage de l'arbre de décision")
    pruning = experiment.experiment_pruning()
    print(pruning.to_string(index=False))
    print(f"[durée de l'étape : {time.perf_counter() - start:.1f}s]")

    if not fast:
        start = _stage("ÉTAPE 6  Early stopping du gradient boosting")
        es = experiment.experiment_early_stopping()
        print(es)
        print(f"[durée de l'étape : {time.perf_counter() - start:.1f}s]")

    print("\n" + "=" * 78)
    print("COMPARAISON DES MODÈLES (triée par macro-F1 test)")
    print("=" * 78)
    cols = ["vectorizer", "model", "cv_f1_macro", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    print(results[cols].to_string(index=False))
    print(f"\nDurée totale : {time.perf_counter() - t0:.1f}s")
    print(f"Artefacts écrits dans : {config.RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="saute l'expérience la plus lente")
    main(**vars(parser.parse_args()))
