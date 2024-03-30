import unittest
from housingpriceprediction.train import train_linear_regression
import numpy as np

class TestTrainLinearRegression(unittest.TestCase):
    
    def test_train_linear_regression(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        y_train = np.array([2, 3, 4])
        model = train_linear_regression(X_train, y_train)
        
if __name__ == '__main__':
    unittest.main()