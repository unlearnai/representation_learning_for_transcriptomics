import logging
import numpy as np
import pandas as pd

from representation_learning_for_transcriptomics.supervised import ProportionalHazardRegressor_lfl as PHR
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

def maybe_int(x):
    try:
        return int(x)
    except ValueError:
        return x


def run_cph():
    """
    Runs a double-CV evaluation of a Cox proportional hazards model.  This example
    uses Lifelines' Newton-Raphson-based fitter which will take maybe ~5 hours runtime.

    Args:
        None

    Returns:
        None

    """
    with pd.HDFStore('./OT_clr_train_STAD_PFI.h5', 'r') as store:
        X = store['expression'].values
        C = store['censor_labels'].values
        Y = store['labels'].values
        # stack labels and censor mask into a single tensor
        Y = np.hstack((Y.reshape(-1,1), C.reshape(-1,1)))

    # standardize expression
    mu = np.mean(X,axis=0)
    std = np.std(X, axis=0)
    X = (X-mu)/std

    # define CVmodel object to manage hyperparameter selection
    cvmodel = CVmodel(PHR, [1e-5,1e-4,1e-3,1e-2], 'penalizer')

    # define Predictor object to handle double CV
    cph_predictor = Predictor(cvmodel, scorers.C_index_scorer)

    # cross validate
    num_folds = 5
    cph_cross_validation_scores = \
        cph_predictor.cross_validate(X, Y,
                                     outer_folds=num_folds, inner_folds=num_folds,
                                     stratified=False, step_size=0.2,
                                     show_progress=False)
    logger.info('Cox PH predictor = {0:.3f}'.format(
                np.mean(cph_cross_validation_scores)))


if __name__ == '__main__':
    run_cph()

