import argparse
import logging
import mlflow

from housingpriceprediction.ingest_data import (
    fetch_housing_data,
    load_housing_data,
    prepare_data_for_training,
)

from housingpriceprediction.logging import ingest_logging
def main():
    ingest_logging()
    parser = argparse.ArgumentParser(
        description="Fetch and prepare data for training."
    )
    parser.add_argument(
        "--raw_path", type=str, default="data/raw",
        help="Path to save raw data."
    )
    parser.add_argument(
        "--processed_path", type=str, default="data/processed",
        help="Path to save processed data."
    )
    args = parser.parse_args()

    fetch_housing_data(raw_path=args.raw_path)
    housing = load_housing_data(raw_path=args.raw_path)
    X_train, X_test, y_train, y_test = prepare_data_for_training(
        housing, processed_path=args.processed_path
    )

    X_train.to_csv(f"{args.processed_path}/X_train.csv", index=False)
    X_test.to_csv(f"{args.processed_path}/X_test.csv", index=False)
    y_train.to_csv(f"{args.processed_path}/y_train.csv", index=False)
    y_test.to_csv(f"{args.processed_path}/y_test.csv", index=False)
    logging.info("Data processing and saving completed.")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_artifact(args.raw_path)
    mlflow.log_artifact(args.processed_path)


if __name__ == "__main__":
    main()