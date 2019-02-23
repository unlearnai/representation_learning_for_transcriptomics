from lifelines import CoxPHFitter
import pandas as pd

class ProportionalHazardRegressor_lfl(object):
    """
    Thin wrapper on Lifelines' cox proportional hazards fitter to be used with
    CVmodel.

    Attributes:
        model_kwargs (dict): keyword arguments to pass to CoxPHFitter's constructor
        model (CoxPHFitter): lifelines.CoxPHFitter object

    """
    def __init__(self, **kwargs):
        """
        Constructs a ProportionalHazardRegressor_lfl object.

        An lifelines.CoxPHFitter instance is created when
        the fit method is called.

        Args:
            kwargs (dict): kwargs to pass to the constructor of
                lifelines.CoxPHFitter.

        Returns:
            ProportionalHazardRegressor_lfl

        """
        self.model_kwargs = kwargs
        self.model = None

    def fit(self, X_train, Y_train, X_validate=None, Y_validate=None,
            cv_param=0.0, **fit_kwargs):
        """
        Create and fit the ProportionalHazards_lfl.
        The cv_param is the l2-penalizer term accepted CoxPHFitter.

        Args:
            X_train (numpy.ndarray ~ (num_samples, num_units)): training data.
            Y_train (numpy.ndarray ~ (num_samples,)): training labels.
            X_validate (numpy.ndarray ~ (num_samples, num_units)):
                validation data. Unused for this model.
            Y_validate (numpy.ndarray ~ (num_samples,)):
                validation labels. Unused for this model.
            cv_param: the value of the hyperparameter optimized in CV.
                The l2 penalizer term.
            fit_kwargs (dict): kwargs to pass to the fit method.

        Returns:
            None

        """
        self.model = CoxPHFitter(penalizer=cv_param, **self.model_kwargs)
        y = pd.DataFrame(Y_train, columns=['time','censor'])
        df = pd.concat((pd.DataFrame(X_train), y), axis=1)
        self.model.fit(df, 'time', 'censor', **fit_kwargs)

    def predict(self, X):
        """
        Predict survival expectations for X.

        Args:
            X (numpy.ndarray ~ (num_samples, num_features)): samples to predict
                survival times from.

        Returns:
            times (numpy.ndarray ~ (num_samples, num_classes)):
                expected survival times.

        """
        assert self.model is not None, "Need to fit model first"
        return self.model.predict_expectation(X)
