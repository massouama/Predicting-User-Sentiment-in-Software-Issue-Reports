"""Package source du projet de classification de sentiment des avis.

Modules
-------
``real_data``      chargement et mise en cache du corpus d'avis Facebook
``preprocessing``  nettoyage / tokenisation / lemmatisation du texte
``vectorization``  vectoriseurs BoW, TF-IDF, Word2Vec (et Word2Vec pré-entraîné)
``balancing``      SMOTE et sous-échantillonnage pour le déséquilibre
``models``         fabriques de classifieurs et grilles d'hyper-paramètres
``evaluation``     métriques et fonctions de figures
``experiment``     orchestration de bout en bout des expériences
"""
