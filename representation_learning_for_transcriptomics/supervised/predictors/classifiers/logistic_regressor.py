import logging
from sklearn.linear_model import LogisticRegression as LR

from ....log import logger

class LogisticRegressor_skl(object):
    """
    A thin wrapper on sklearn.linear_model.LogisticRegression, used to interface
    with CVmodel.

    Attributes:
        model_kwargs (dict): keyword arguments to pass to LogisticRegression constructor
        model (LogisticRegression): sklearn.linear_model.LogisticRegression object

    """
    def __init__(self, **kwargs):
        """
        Constructs a LogisticRegression_skl object.

        An sklearn.linear_model.LogisticRegression instance is created when
        the fit method is called.

        Args:
            kwargs (dict): kwargs to pass to the constructor of
                sklearn.linear_model.LogisticRegression.

        Returns:
            LogisticRegression_skl

        """
        self.model_kwargs = kwargs
        self.model = None

    def fit(self, X_train, Y_train, X_validate=None, Y_validate=None,
            cv_param=1.0, **fit_kwargs):
        """
        Create and fit the LogisticRegression_skl.
        The cv_param is the 1/C, the l2 penalty coefficient.

        Args:
            X_train (numpy.ndarray ~ (num_samples, num_units)): training data.
            Y_train (numpy.ndarray ~ (num_samples,)): training labels.
            X_validate (numpy.ndarray ~ (num_samples, num_units)):
                validation data. Unused for this model.
            Y_validate (numpy.ndarray ~ (num_samples,)):
                validation labels. Unused for this model.
            cv_param: the value of the hyperparameter optimized in CV.
                The l2 penalty coefficient.
            fit_kwargs (dict): kwargs to pass to the fit method.

        Returns:
            None

        """
        self.model = LR(C = 1.0/cv_param,
                         verbose=(logger.isEnabledFor(logging.SUPERVISED)),
                         **self.model_kwargs)
        self.model.fit(X_train, Y_train, **fit_kwargs)

    def predict(self, X):
        """
        Predict class probabilities from data.

        Args:
            X (numpy.ndarray ~ (num_samples, num_features)): samples to predict
                class probabilities from.

        Returns:
            probabilities (numpy.ndarray ~ (num_samples, num_classes)):
                class probabilities.

        """
        assert self.model is not None, "Need to fit model first"
        return self.model.predict_proba(X)

    def predict_class(self, X):
        """
        Predict class labels from data.

        Args:
            X (numpy.ndarray ~ (num_samples, num_features)): samples to predict
                class labels from.

        Returns:
            labels (numpy.ndarray ~ (num_samples,)): class labels.

        """
        assert self.model is not None, "Need to fit model first"
        return self.model.predict(X)