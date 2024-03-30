import os
import unittest
import pandas as pd

from housingpriceprediction import ingest_data


class test_data_ingestion(unittest.TestCase):
    def test_fetch_housing_data(self):
        ingest_data.fetch_housing_data()
        self.assertTrue(os.path.exists("data/housing/housing.csv"))