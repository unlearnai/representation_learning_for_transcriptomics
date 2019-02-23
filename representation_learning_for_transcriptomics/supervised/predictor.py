import numpy as np
from pathlib import Path
import pickle

from cytoolz import identity

from .predictors import common
from ..log import logger


class Predictor(object):
    """
    Abstract predictor class which can manage scoring estimation via cross validation.

    Attributes:
        predictor (predictor object): underlying predictor object, e.g. CVModel
        scorer (callable): (X_test,Y_test,model)->score, a scoring function
        x_transform (callable): preprocessing transform on X data field
        y_transform (callable): preprocessing transform on Y data field

    """
    def __init__(self, predictor, scorer, x_transform=identity,
                 y_transform=identity):
        """
        Class to manage fitting and analysis of supervised models.

        Args:
            predictor (object): specific ./predictors/ class for predictive models
            scorer (callable): a function to compute a model score
            x_transform (optional; callable): function to apply to the
                independent variables before sending them through the predictor
            y_transform (optional; callable): function to apply to the
                dependent variables before sending them through the predictor

        Returns:
            Predictor

        """
        self.predictor = predictor
        self.scorer = scorer
        self.x_transform = x_transform
        self.y_transform = y_transform

    @classmethod
    def load(cls, filename):
        """
        Create a predictor from a saved object.

        Args:
            filename (str)

        Returns:
            Predictor

        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filepath, overwrite_existing=False):
        """
        Save the predictor object for future use at the given filepath.

        Args:
            filepath (string): absolute filepath to save file
            overwrite_existing (bool): whether or not to overwrite existing modelfile

        Returns:
            None

        """
        path = Path(filepath)
        assert overwrite_existing or not path.exists(), "Must allow overwriting existing files"

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def evaluate(self, data, labels):
        """
        Evaluate the predictor.

        Args:
            data (numpy.ndarray): the matrix of features (n_samples, n_features)
            labels (numpy.ndarray): the vector of labels (n_samples,)

        Returns:
            score (float)

        """
        X = self.x_transform(data)
        Y = self.y_transform(labels)
        return self.scorer(self.predictor, X, Y)

    def predict(self, data):
        """
        Predict labels.

        Args:
            data (numpy.ndarray): the matrix of features (n_samples, n_features)

        Returns:
            predictions (numpy.ndarray): the vector of predictions (nsamples,)

        """
        X = self.x_transform(data)
        return self.predictor.predict(X)

    def fit(self, data, labels, nfolds=5, stratified=False, **fit_kwargs):
        """
        Fit the predictor.

        Args:
            data (numpy.ndarray): the matrix of features (n_samples, n_features)
            labels (numpy.ndarray): the vector of labels (n_samples,)
            nfolds (int): number of data folds to use in predictor.fit
            stratified (bool): whether or not to use stratified KFold or not
            fit_kwargs (kwargs): keyword arguments to pass to predictor.fit

        Returns:
            None

        """
        X = self.x_transform(data)
        Y = self.y_transform(labels)
        self.predictor.fit(X, Y, scorer=self.scorer,
                           nfolds=nfolds, stratified=stratified, **fit_kwargs)

    def cross_validate(self, data, labels, outer_folds=5, inner_folds=5,
                       stratified=False, **fit_kwargs):
        """
        Estimate the performance of the predictor using cross-validation.

        Args:
            data (numpy.ndarray): the matrix of features (n_samples, n_features)
            labels (numpy.ndarray): the vector of labels
            outer_folds (optional; int): the number of folds to use for score assessment
            inner_folds (optional; int): the number of folds to use for hyperparameter
                selection
            stratified (bool): whether to use sklearn's KFold or StratifiedKFold
                for folding data in cross-validation loops
            fit_kwargs (kwargs): keyword arguments to pass to predictor.fit

        Returns:
            errors (numpy.ndarray)

        """
        errors = np.zeros((outer_folds,))
        folds = common.create_cv_folds(outer_folds, stratified)
        i = 0
        logger.predictor("Cross validating with {} outer folds, and {} "
                         "inner folds".format(outer_folds, inner_folds))
        for train_index, test_index in folds.split(data, labels):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            self.fit(X_train, y_train, nfolds=inner_folds, stratified=stratified,
                     **fit_kwargs)
            errors[i] = self.evaluate(X_test, y_test)
            i += 1
            logger.predictor("Fold {} error: {}".format(i, errors[i-1]))

        return errors
