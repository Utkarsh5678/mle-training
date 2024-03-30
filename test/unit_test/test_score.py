import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from housingpriceprediction.score import score_model_rmse, score_model_mae, RF_score


class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        param_grid = {'n_estimators': [10, 20, 30], 'max_depth': [None, 5, 10]}
        rf = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)
        self.cvres = grid_search.cv_results_
        self.model = RandomForestRegressor(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_score_model_rmse(self):
        rmse = score_model_rmse(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(rmse, 0)
        
    def test_score_model_mae(self):
        mae = score_model_mae(self.model, self.X_test, self.y_test)
        self.assertGreaterEqual(mae, 0)

    

if __name__ == '__main__':
    unittest.main()