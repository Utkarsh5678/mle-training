import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def score_model_rmse(model, X_test, y_test):
    """
    Calculate the root mean squared error (RMSE) of a model.
    Parameters:
    model: The trained model for prediction.
    X_test: The input features for testing.
    y_test: The actual target values for testing.
    Returns:
    float: The root mean squared error (RMSE) of the model predictions.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse


def score_model_mae(model, X_test, y_test):
    """
    Calculate the Mean Absolute Error (MAE) of the model predictions.
    Parameters:
    - model: the trained model for prediction
    - X_test: the input features for testing
    - y_test: the actual target values for testing
    Returns:
    - mae: the Mean Absolute Error (MAE) between the predicted values and the actual values
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    return mae


def RF_score(cvres):
    """
    Prints the root mean squared error (RMSE) score and the corresponding parameters for each cross-validation fold.
    Parameters:
    - cvres (dict): A dictionary containing the cross-validation results. It should have the following keys:
        - "mean_test_score" (array-like): An array-like object containing the mean test scores for each fold.
        - "params" (array-like): An array-like object containing the corresponding parameters for each fold.
    Returns:
    - None
    """
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)