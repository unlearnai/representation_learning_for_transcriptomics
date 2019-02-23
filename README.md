# representation learning for transcriptomics

This code accompanies the paper by Smith et. al.,
[*Deep learning of representations for transcriptomics-based phenotype prediction.*](https://www.unlearn.health)

This module contains a small python-based framework for implementing
 doubly-nested cross validation. The module contains three examples demonstrating
 the use of nested cross validation to evaluate performance of various prediction models on transcriptomics data.
 One predicts a binary label using ridge regression as well as a random forest predictor.  A second example uses Cox proportional hazards for cancer survival prediction from gene expression data.
 All of the examples of classifiers and regressors in this code dump are wrappers from
 open source libraries such as [sci-kit learn](https://scikit-learn.org/stable/) and [lifelines](https://lifelines.readthedocs.io/en/latest/).

Modules:
- log: logging utilities for the library
- supervised
    - predictors
        - classifiers
            + logistic_regressor.py: wrapper around sklearn's logistic regression classifier
            + random_forest_classifier.py: wrapper around sklearn's random forest classifier
        - regressors
            + proportional_hazard_regressor.py: wrapper around lifelines' proportional hazard regressor
        + scorers.py: functions that compute a metric from an sklearn classifier and data
        + common.py: utilities used in multiple supervised models
    + predictor.py: wrapper to perform the outer loop of nested cross-validation

To get started, install with package with pip:
Run `pip install -e .` from the root directory of the repository.
