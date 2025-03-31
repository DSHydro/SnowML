""" Module to calculate various dianostic metrics """

import numpy
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error

def kling_gupta_efficiency(y_true, y_pred):
    """
    Calculate the Kling-Gupta Efficiency (KGE) and its components.

    The KGE is a metric used to evaluate the performance of hydrological models.
    It is composed of three components: correlation (r), variability ratio 
    (alpha), and bias ratio (beta).

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    tuple: A tuple containing the following elements:
        - kg (float): Kling-Gupta Efficiency.
        - r (float): Correlation coefficient between y_true and y_pred.
        - alpha (float): Variability ratio (standard deviation of y_pred / 
            standard deviation of y_true).
        - beta (float): Bias ratio (mean of y_pred / mean of y_true).

    Notes:
    - If NaN values are detected in y_true or y_pred, the function will print 
        an error message and return (np.nan, np.nan, np.nan, np.nan).
    - If zero variance is detected in y_true or y_pred, the function will print 
        an error message and return (np.nan, np.nan, np.nan, np.nan).

    Example:
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    >>> kling_gupta_efficiency(y_true, y_pred)
    (0.993, 0.998, 1.0, 1.02)
    """
    # Check for NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Error: NaN values detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    # Check for zero variance
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        print("Error: Zero variance detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    # Compute correlation
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]

    # Compute KGE components
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)
    kg = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    print(f"r: {r}, alpha: {alpha}, beta: {beta}")
    return kg, r, alpha, beta


def calc_metrics(d_true, d_pred, type = "test"): 
    metrics = [mse, kge, r2, mae]
    metric_names = [f"{type}_metric" for metric in metrics]
    r2 = r2_score(d_true, d_prod)
    kge, _, _, _ = kling_gupta_efficiency(d_true, d_pred)
    mse = mean_squared_error(d_true, d_prod)
    mae = mean_squared_error(d_true, d_prod)
    metric_dict = dict(zip(metric_names, [mse, kge, r2, mae]))
    return metric_dict
