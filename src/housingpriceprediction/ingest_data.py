# import os
# import tarfile
# import urllib.request

# import numpy as np
# import pandas as pd
# from sklearn.impute import SimpleImputer
# from sklearn.model_selection import StratifiedShuffleSplit
import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = "data"
RAW_PATH = os.path.join(DATA_PATH, "raw")
PROCESSED_PATH = os.path.join(DATA_PATH, "processed")
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, raw_path=RAW_PATH):
    os.makedirs(raw_path, exist_ok=True)
    tgz_path = os.path.join(raw_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=raw_path)
    housing_tgz.close()


def load_housing_data(raw_path=RAW_PATH):
    csv_path = os.path.join(raw_path, "housing.csv")
    return pd.read_csv(csv_path)


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 0, 1, 2, 3
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        return np.c_[X, rooms_per_household, bedrooms_per_room, population_per_household]



def prepare_data_for_training(housing, processed_path=PROCESSED_PATH):
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    os.makedirs(processed_path, exist_ok=True)
    # Define numerical features
    num_features = ['longitude','latitude','housing_median_age','total_rooms', 'total_bedrooms', 'population', 'households','median_income']

    # Define categorical features
    cat_features = ['ocean_proximity']

    # Pipeline for numerical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('feature_engineering', FeatureEngineeringTransformer()),
        ('std_scaler', StandardScaler())
    ])

    # Pipeline for categorical features
    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder())
    ])

    # Full pipeline
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"].copy()

    X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_prepared = pd.DataFrame(full_pipeline.transform(X_test))

    return X_train_prepared, X_test_prepared, y_train, y_test