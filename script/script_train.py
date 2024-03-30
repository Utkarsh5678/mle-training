import argparse
import os
import joblib
import pandas as pd

from housingpriceprediction import ingest_data
from housingpriceprediction import train


def main(args):
    # Load dataset
    df = {}
    files = os.listdir(input_dr)
    for f in files:
        file_path = os.path.join(input_dr, f)
        if not file_path.endswith(".csv"):
            continue  # Skip non-CSV files
        try:
            df[f] = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue  # Skip empty CSV files

    housing = df["housing.csv"]

    train_set, test_set = ingest_data.stratified_split(housing)
    housing, housing_labels = ingest_data.explore_housing_data(
        housing, train_set, test_set
    )  # noqa
    housing_prepared = ingest_data.preprocess_data(housing)

    # Train models
    linear_reg_model = train.train_linear_regression(
        housing_prepared, housing_labels
    )  # noqa
    decision_tree_model = train.train_decision_tree(
        housing_prepared, housing_labels
    )  # noqa
    random_forest_model = train.train_random_forest(
        housing_prepared, housing_labels
    )  # noqa
    tuned_random_forest_model = train.tune_random_forest(
        housing_prepared, housing_labels
    )  # noqa
    final_model = tuned_random_forest_model.best_estimator_

    # Save models
    joblib.dump(linear_reg_model, args.output_dr + "/linear_reg_model.pkl")
    joblib.dump(
        decision_tree_model, args.output_dr + "/decision_tree_model.pkl"
    )  # noqa
    joblib.dump(
        random_forest_model, args.output_dr + "/random_forest_model.pkl"
    )  # noqa
    output_path = args.output_dr + "/tuned_random_forest_model.pkl"
    joblib.dump(tuned_random_forest_model, output_path)
    joblib.dump(final_model, args.output_dr + "/final_model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on input directory and save them to output directory"  # noqa
    )
    parser.add_argument(
        "input_dr", type=str, help="Path to the dataset directory"
    )  # noqa
    parser.add_argument(
        "output_dr", type=str, help="Path to the output directory"
    )  # noqa
    args = parser.parse_args()
    input_dr = args.input_dr
    output_dr = args.output_dr
    if not os.path.exists(output_dr):
        os.makedirs(output_dr)

    main(args)
# python script/train.py data/raw artifacts/model # noqa