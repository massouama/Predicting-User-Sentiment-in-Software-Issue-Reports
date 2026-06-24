"""main.py — Point d'entrée : lance toute l'étude en séquence.

Usage :
    python main.py          # exécution complète (~8 min sur 4 cœurs)
    python main.py --rapide # saute l'expérience d'early stopping

Toutes les sorties (tableaux CSV, matrices de confusion, figures)
sont écrites dans results/ à la racine du dépôt.
"""
import argparse
import sys
import time
from pathlib import Path

# Ajoute la racine du dépôt au chemin Python pour que les imports fonctionnent
# depuis ce sous-dossier.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.experiences import (
    comparaison_principale,
    experience_desequilibre,
    experience_early_stopping,
    experience_elagage,
    experience_pca,
)


def _titre(texte: str) -> float:
    """Affiche un titre de section et renvoie l'heure de début."""
    print("\n" + "=" * 70)
    print(texte)
    print("=" * 70)
    return time.perf_counter()


def main(rapide: bool = False) -> None:
    # Crée les dossiers de sortie s'ils n'existent pas.
    config.creer_dossiers()
    debut = time.perf_counter()

    # ── Étape 1-2 : données + comparaison principale ──────────────────────────
    t = _titre("ÉTAPE 1-2  Chargement des données + comparaison vectoriseur × classifieur")
    resultats = comparaison_principale()
    meilleur = resultats.iloc[0]
    print(f"\nMeilleur modèle : {meilleur['vectoriseur']} + {meilleur['modele']} "
          f"(test macro-F1 = {meilleur['f1_macro']:.3f})")
    print(f"[durée : {time.perf_counter() - t:.1f}s]")

    # ── Étape 3 : déséquilibre ────────────────────────────────────────────────
    t = _titre("ÉTAPE 3  Gestion du déséquilibre (aucun / SMOTE / sous-échantillonnage)")
    deseq = experience_desequilibre()
    print(deseq.to_string(index=False))
    print(f"[durée : {time.perf_counter() - t:.1f}s]")

    # ── Étape 4 : réduction de dimension ──────────────────────────────────────
    t = _titre("ÉTAPE 4  Réduction de dimension (TruncatedSVD / PCA)")
    pca = experience_pca()
    print(pca.to_string(index=False))
    print(f"[durée : {time.perf_counter() - t:.1f}s]")

    # ── Étape 5 : élagage ─────────────────────────────────────────────────────
    t = _titre("ÉTAPE 5  Élagage de l'arbre de décision")
    elagage = experience_elagage()
    print(elagage.to_string(index=False))
    print(f"[durée : {time.perf_counter() - t:.1f}s]")

    # ── Étape 6 : early stopping (optionnelle) ────────────────────────────────
    if not rapide:
        t = _titre("ÉTAPE 6  Early stopping (Gradient Boosting)")
        es = experience_early_stopping()
        print(es)
        print(f"[durée : {time.perf_counter() - t:.1f}s]")

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RÉSULTATS — classés par macro-F1 test décroissant")
    print("=" * 70)
    colonnes = ["vectoriseur", "modele", "cv_f1_macro",
                "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    print(resultats[colonnes].to_string(index=False))
    print(f"\nDurée totale : {time.perf_counter() - debut:.1f}s")
    print(f"Résultats dans : {config.RESULTATS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rapide", action="store_true",
                        help="saute l'expérience d'early stopping (plus lente)")
    main(**vars(parser.parse_args()))
