import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold

def create_cv_folds(n_folds, stratified, shuffle=True):
    """
    Create a KFold  object from sklearn.
    Wraps the construction of xtratified or regular KFolds.

    Args:
        n_folds (int): the number of cross validation folds
        stratified (bool): whether or not to use stratified cross validation
        shuffle (optional; bool): whether or not to shuffle the examples

    Returns:
        sklearn KFolds-like object

    """
    if stratified:
        return StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    return KFold(n_splits=n_folds, shuffle=shuffle)

def class_labels_to_one_hot(class_labels, num_classes):
    """
    Convert integral class label vector to a one-hot tensor.

    Args:
        class_labels (numpy.ndarray[int] ~ (num_samples,))
        num_classes (int): the number of classes

    Returns:
        numpy.ndarray ~ (num_samples, num_classes)

    """
    class_labels_int = class_labels.astype(np.int)
    return np.identity(num_classes, dtype=np.float32)[class_labels_int]
