from . import common
from ...log import logger

class CVmodel(object):
    """
    Performs the CV loop of a model over hyperparameters.

    Provides methods to optimize the model over a set of hyperparameters and
    to make predictions from the optimized model.

    Attributes:
        model_type (type): model type, e.g. classifiers.LogisticRegressor
        model_kwargs (dict): kwargs to pass to model constructor
        cv_params (list): list of parameter values to cross validate over
        cv_param_name (str): name of parameter to cross validate over
        cv_model (model_type): best model fit with params chosen by crossvalidation
        cv_hyperparams (dict): final chosen hyperparameters for cv_model

    """
    def __init__(self, model_type, cv_param_list, cv_param_name, **model_kwargs):
        """
        Constructs a CVmodel object.

        Args:
            model_type (type): the type of a model object.
            cv_param_list (List): hyperparameter values to use in the CV.
            cv_param_name (str): the name of the hyperparameter being selected.
            model_kwargs (dict): kwargs to pass to the underlying model constructor.

        Returns:
            CVmodel

        """
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.cv_params = cv_param_list
        self.cv_param_name = cv_param_name
        self.cv_model = None
        self.cv_hyperparams = {}

    def fit(self, X, Y, nfolds, scorer, maximize_score=True, stratified=True,
            **fit_kwargs):
        """
        Run the CV loop to find the optimal hyperparameter from a list.
        Once the optimal hyperparameter is found, the model is fit on the
        entire dataset with this hyperparameter.

        Args:
            X (numpy.ndarray ~ (num_samples, num_units)): data.
            Y (numpy.ndarray ~ (num_samples, label_dim)): labels.
            nfolds (int): the number of folds to use for cross-validaiton.
            scorer (callable(model, X, Y) -> float): scoring function.
            maximize_score (optional; bool): whether to maximize (or minimize)
                the score.
            stratified (optional; bool): whether or not to use stratified CV.
            fit_kwargs (dict): kwargs to pass to the fit method.

        Returns:
            None

        """
        best_score = -1e6 if maximize_score else 1e6
        best_cv_param = None
        # loop over the hyperparameters to select through CV
        for cv_param in self.cv_params:
            logger.cv("trying {} = {}".format(self.cv_param_name, cv_param))
            mean_score = 0

            # create the model
            model = self.model_type(**self.model_kwargs)
            # create the CV folds
            folds = common.create_cv_folds(nfolds, stratified)

            # CV: loop over each fold
            for train_index, test_index in folds.split(X, Y):
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                model.fit(X_train, Y_train, X_test, Y_test,
                          cv_param=cv_param, **fit_kwargs)
                mean_score += scorer(model, X_test, Y_test)
            mean_score /= nfolds
            logger.cv("score = {}".format(mean_score))
            if (maximize_score and mean_score > best_score) or \
               (not maximize_score and mean_score < best_score):
                best_score = mean_score
                best_cv_param = cv_param

        # final fit with best hyperparameters
        self.cv_model = self.model_type(**self.model_kwargs)
        self.cv_model.fit(X, Y, cv_param=best_cv_param, **fit_kwargs)
        logger.cv("Refit with best CV parameter {} which achieved score {}"
                    .format(best_cv_param, best_score))
        self.cv_hyperparams = {self.cv_param_name: best_cv_param, **fit_kwargs}

    def predict(self, X):
        """
        Predicts outputs from the trained model, given X.  For a classifier,
        this is the class probabilities.

        Args:
            X (numpy.ndarray ~ (num_samples, num_features)): samples to predict
                outputs from.

        Returns:
            predicted labels (numpy.ndarray): predicted outputs for each sample
                in X.  For a classifier, the class probabilities.

        """
        assert self.cv_model is not None, "Need to fit model first"
        return self.cv_model.predict(X)

    def predict_class(self, X):
        """
        Predict class labels from X.  Will raise a NotImplementedError for
        non-classifier models.

        Args:
            X (numpy.ndarray ~ (num_samples, num_features)): samples to predict
                class labels from.

        Returns:
            probabilities (numpy.ndarray ~ (num_samples,)): predicted class
                for each element in X.

        """
        assert self.cv_model is not None, "Need to fit model first"
        try:
            return self.cv_model.predict_class(X)
        except AttributeError:
            raise NotImplementedError("{} does not allow class prediction!".format(
                                        self.model_type.__name__))
