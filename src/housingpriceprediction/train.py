from joblib import dump
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
import os

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model using the provided training data.
    Parameters:
    X_train (array-like): The input features for training.
    y_train (array-like): The target values for training.
    Returns:
    lin_reg (LinearRegression): The trained linear regression model.
    """
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    return lin_reg


def train_decision_tree(X_train, y_train):
    """
    Trains a decision tree regressor model on the given training data.
    Parameters:
        X_train (array-like): The input features of the training data.
        y_train (array-like): The target values of the training data.
    Returns:
        DecisionTreeRegressor: The trained decision tree regressor model.
    """

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(X_train, y_train)
    return tree_reg


def rand_tune_random_forest(X_train, y_train):
    """
    Generate a random forest regressor model using randomized search.
    Parameters:
    - X_train: The training input samples.
    - y_train: The target values.
    Returns:
    - rnd_search: A random forest regressor model fitted using randomized search.
    """
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(X_train, y_train)
    return rnd_search


def grid_tune_random_forest(X_train, y_train):
    """
    A function that performs grid search with random forest regressor.
    Parameters:
    - X_train: the training data
    - y_train: the target values
    Returns:
    - grid_search: the grid search object with the best parameters
    """
    param_grid = [
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        {
            "bootstrap": [False],
            "n_estimators": [3, 10],
            "max_features": [2, 3, 4],
        },
    ]
    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search