import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from lifelines.utils import concordance_index

from . import common

def roc_auc_scorer(model, X, Y):
    """
    Scoring function built out of the roc auc score.

    Args:
        model (classifier): model to eval
        X (numpy.ndarray): data to predict from
        Y (numpy.ndarray): true labels

    Returns:
        ROC AUC (float)

    """
    # sklearn uses predict_proba to get class probabilities, use that if available
    try:
        preds = model.predict_proba(X)
    except AttributeError:
        preds = model.predict(X)
    Y_classhot = common.class_labels_to_one_hot(Y, preds.shape[1])
    return roc_auc_score(Y_classhot, preds)


def accuracy_scorer(model, X, Y):
    """
    Scoring function built out of the accuracy score.

    Args:
        model (sklearn classifier): model to eval
        X (numpy.ndarray): data to predict from
        Y (numpy.ndarray): true labels

    Returns:
        accuracy (float)

    """
    # sklearn uses predict_proba to get class probabilities, use that if available
    try:
        preds = model.predict_proba(X)
    except AttributeError:
        preds = model.predict(X)
    return accuracy_score(Y, np.argmax(preds, axis=1))


def r2_scorer(model, X, Y):
    """
    Scoring function built out of the r2 score.

    Args:
        model (sklearn classifier): model to eval
        X (numpy.ndarray): data to predict from
        Y (numpy.ndarray): true labels

    Returns:
        R2 score (float)

    """
    return r2_score(Y, model.predict(X))


def rmse_scorer(model, X, Y):
    """
    Scoring function built from the root mean squared error.

    Args:
        model (sklearn classifier): model to eval
        X (numpy.ndarray): data to predict from
        Y (numpy.ndarray): true labels

    Returns:
        root mean squared error (float)

    """
    return np.sqrt(mean_squared_error(Y, model.predict(X)))


def C_index_scorer(model, X, Y):
    """
    Scoring function built from C-index.  Y must have the form of a matrix containing
    two columns, the time of the event and a censoring indicator.

    Args:
        model (sklearn classifier): model to eval
        X (numpy.ndarray): data to predict from
        Y (numpy.ndarray): true labels

    Returns:
        C-index of lifetime predictions

    """
    predicted = model.predict(X)
    return concordance_index(Y[:,0], predicted, Y[:,1])
