import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Fetches housing data from a given URL and saves it to a specified path.
    Args:
        housing_url (str): The URL from which to fetch the housing data. Defaults to HOUSING_URL.
        housing_path (str): The path to which the housing data should be saved. Defaults to HOUSING_PATH.
    Returns:
        None
    """
    os.makedirs(housing_path, exist_ok=True)  
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

# Rest of the code remains the same



def load_housing_data(housing_path=HOUSING_PATH):
    """
    Load housing data from a specified path.
    Parameters:
    housing_path (str): The path to the directory containing the housing data.
    Returns:
    pandas.DataFrame: A DataFrame containing the loaded housing data.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def prepare_data_for_training(housing):
    """
    Prepare the housing data for training by performing the following steps:
    
    1. Create a new column 'income_cat' in the housing dataframe, which categorizes the 'median_income' column into 5 categories.
    2. Split the housing dataframe into a stratified train and test set using the 'income_cat' column.
    3. Drop the 'income_cat' column from both the train and test sets.
    4. Separate the numerical and categorical features from the train and test sets.
    5. Impute missing values in the numerical features using the 'median' strategy.
    6. Calculate additional features based on the numerical features: 'rooms_per_household', 'bedrooms_per_room', and 'population_per_household'.
    7. One-hot encode the categorical features in the train set.
    8. Impute missing values in the numerical features of the test set using the same imputer fitted on the train set.
    9. Calculate additional features based on the numerical features of the test set.
    10. One-hot encode the categorical features in the test set.
    11. Return the prepared train and test sets, as well as the target variables for training.
    """
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

    X_train = strat_train_set.drop("median_house_value", axis=1)
    y_train = strat_train_set["median_house_value"].copy()


    num_attribs = list(X_train.drop(["ocean_proximity"],axis=1))
    cat_attribs = ["ocean_proximity"]
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
        ])
    from sklearn.compose import ColumnTransformer


    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
        ])

    X_train_prepared = pd.DataFrame(full_pipeline.fit_transform(X_train))
    
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    X_test_prepared = pd.DataFrame(full_pipeline.fit_transform(X_test))





    return X_train_prepared, X_test_prepared, y_train, y_test
