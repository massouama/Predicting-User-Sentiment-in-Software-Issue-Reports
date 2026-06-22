"""Source package for the issue-report sentiment-classification project.

Modules
-------
``data_generation``  reproducible synthetic issue-report corpus
``preprocessing``    text cleaning / tokenisation / lemmatisation
``vectorization``    BoW, TF-IDF, Word2Vec and (optional) BERT vectorisers
``balancing``        SMOTE and random under-sampling for class imbalance
``models``           classifier factories and hyper-parameter grids
``evaluation``       metrics and figure helpers
``experiment``       end-to-end orchestration of the experiments
"""
