import os
import unittest
import pandas as pd

from housingpriceprediction import ingest_data


class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        # Fetch housing data before each test
        ingest_data.fetch_housing_data()

    def test_prepare_data_for_training(self):
        housing = ingest_data.load_housing_data()
        X_train_prepared, X_test_prepared, y_train, y_test = ingest_data.prepare_data_for_training(housing)

        # Check if X_train_prepared and X_test_prepared are not None
        self.assertIsNotNone(X_train_prepared)
        self.assertIsNotNone(X_test_prepared)

        # Check if y_train and y_test are not None
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

        # Check if X_train_prepared and X_test_prepared have the expected number of rows and columns
        self.assertEqual(X_train_prepared.shape[0], len(y_train))
        self.assertEqual(X_test_prepared.shape[0], len(y_test))

        # Assert that there are no missing values in X_train_prepared and X_test_prepared
        self.assertTrue(X_train_prepared.isnull().sum().sum() == 0)
        self.assertTrue(X_test_prepared.isnull().sum().sum() == 0)