import argparse
import joblib
import os
import pandas as pd
from housingpriceprediction import ingest_data
from housingpriceprediction import score as scoring


def main(args):
    # Load models
    model_files = os.listdir(args.model_)
    models = {}
    for file in model_files:
        if file.endswith(".pkl"):
            model_name = os.path.splitext(file)[0]
            model = joblib.load(os.path.join(args.model_, file))
            models[model_name] = model

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

    # Score models
    for model_name, model in models.items():
        mae, rmse = scoring.evaluate_model(
            model, housing_prepared, housing_labels
        )  # noqa
        if args.output_mode == "file":
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, "a") as f:
                f.write(f"Model: {model_name}, MAE: {mae}, RMSE: {rmse}\n")
        elif args.output_mode == "print":
            print(f"Model: {model_name}, MAE: {mae}, RMSE: {rmse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score models using the specified dataset"
    )
    parser.add_argument(
        "input_dr", type=str, help="Path to the dataset directory"
    )  # noqa
    parser.add_argument("model_", type=str, help="Path to the model directory")  # noqa
    parser.add_argument(
        "output_mode",
        choices=["file", "print"],
        default="print",
        help="Output mode: 'file' to save to a file, 'print' to print to console (default: 'print')",  # noqa
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output file",
    )
    args = parser.parse_args()
    input_dr = args.input_dr
    model_ = args.model_

    main(args)