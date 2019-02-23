import logging
import numpy as np
import pandas as pd

from representation_learning_for_transcriptomics.supervised import RandomForestClassifier_skl
from representation_learning_for_transcriptomics.supervised import LogisticRegressor_skl
from representation_learning_for_transcriptomics.supervised import CVmodel
from representation_learning_for_transcriptomics.supervised import scorers
from representation_learning_for_transcriptomics.supervised import Predictor
from representation_learning_for_transcriptomics.log import logger

# setup logging
logger.setLevel(logging.CV)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def run_rfc():
    """
    Evaludates the performance of a random forest classifier in predicting a binary label
    on a set of clr-transformed expression data restricted to the OT gene set
    (1530 dimensions).  The inner cv loop selects the max depth parameter of
    the model by 5-fold cross validation.  The outer cv loop gives an n=5 estimate
    of the average performance.

    Args:
        None

    Returns:
        None

    """
    num_folds = 5
    with pd.HDFStore('./OT_clr_train_LGG_grade.h5') as store:
        X = store['expression'].values
        Y = store['labels'].values

    # standardize expression
    mu = np.mean(X,axis=0)
    std = np.std(X, axis=0)
    X = (X-mu)/std

    # define Predictor object to manage nested CV
    rf_predictor = Predictor(
                        CVmodel(RandomForestClassifier_skl,[4,8,16,32,64,128], 'max_depth',
                                n_estimators=100, n_jobs=-1),
                        scorers.accuracy_scorer)
    # cross validate
    rf_cross_validation_scores = \
        rf_predictor.cross_validate(X, Y,
                                    outer_folds=num_folds, inner_folds=num_folds)
    logger.info('Random Forest cross-validation = {0:.3f}'.format(
                np.mean(rf_cross_validation_scores)))

def run_lgr():
    """
    Evaludates the performance of a logistic regression classifier in predicting a binary label
    on a set of clr-transformed expression data restricted to the OT gene set
    (1530 dimensions).  The inner cv loop selects the l_2-weight penalty coefficient
     by 5-fold cross-validation.  The outer cv loop gives an n=5 estimate
    of the average performance.

    Args:
        None

    Returns:
        None

    """
    num_folds = 5
    with pd.HDFStore('./OT_clr_train_LGG_grade.h5') as store:
        X = store['expression'].values
        Y = store['labels'].values

    # standardize expression
    mu = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X-mu)/std

    # define CVmodel to manage hyperparameter selection
    cvmodel = CVmodel(LogisticRegressor_skl,
                      [1e-6, 1e-5, 1e-4, 1e-3, 1e-2,1e-1,1,10,100,1000], 'C^-1',
                      solver = 'lbfgs', max_iter=5000, multi_class='auto')

    # define Predictor object to manage nested CV
    lg_predictor = Predictor(cvmodel,scorers.accuracy_scorer)

    # cross validate
    lg_cross_validation_scores = \
        lg_predictor.cross_validate(X, Y,
                                    outer_folds=num_folds, inner_folds=num_folds)
    logger.info('Logistic Regression cross-validation = {0:.3f}'.format(
                np.mean(lg_cross_validation_scores)))

if __name__ == '__main__':
    run_lgr()
    run_rfc()

