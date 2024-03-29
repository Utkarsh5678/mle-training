import os
import unittest
import pandas as pd

from mypackage import ingest_data


class test_data_ingestion(unittest.TestCase):
    def test_fetch_housing_data(self):
        ingest_data.fetch_housing_data()
        self.assertTrue(os.path.exists("datasets/housing/housing.csv"))