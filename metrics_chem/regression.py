import numpy as np


def error(y_true, y_pred):
    """Compute error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return y_true - y_pred


def relative_error(y_true, y_pred):
    """Compute relative error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return y_true - y_pred


def mean_sqaured_error(y_true, y_pred):
    """Compute mean squared error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return np.average((y_true - y_pred) ** 2, axis=0)


def root_mean_sqaured_error(y_true, y_pred):
    """Compute root mean squared error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return np.sqrt(np.average((y_true - y_pred) ** 2, axis=0))


def mean_error(y_true, y_pred):
    """Compute mean error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return np.average(y_true - y_pred, axis=0)


def mean_absolute_error(y_true, y_pred):    
    """Compute mean absolute error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    return np.average(np.abs(y_true - y_pred), axis=0)


def mean_absolute_percentage_error(y_true, y_pred):
    """Compute mean absolute percentage error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    abs_err = np.abs((y_true - y_pred) / y_true)
    print(abs_err)
    return np.average(abs_err, axis=0)


def mean_sqaured_absolute_percentage_error(y_true, y_pred):
    """Compute mean squared absolute percentage error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    abs_err = np.abs((y_true - y_pred) / y_true)
    return np.average(abs_err ** 2, axis=0)


def root_mean_sqaured_absolute_percentage_error(y_true, y_pred):
    """Compute root mean squared absolute percentage error between ground truth and predicted targets.

    Args:
        y_true (arr): Ground truth
        y_pred (arr): Predicted targets
    """
    abs_err = np.abs((y_true - y_pred) / y_true)
    return np.sqrt(np.average(abs_err ** 2, axis=0))
    