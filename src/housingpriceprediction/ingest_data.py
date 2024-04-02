import os
import tarfile
import urllib.request

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

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

    imputer = SimpleImputer(strategy="median")

    housing_num = X_train.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=X_train.index)

    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = X_train[["ocean_proximity"]]
    X_train_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    return X_train_prepared, X_test_prepared, y_train, y_test
