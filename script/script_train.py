import argparse
import os
import joblib
import pandas as pd
import logging

from housingpriceprediction import ingest_data
from housingpriceprediction import train

def setup_logging(log_file):
    log_folder = "logs"
    os.makedirs(log_folder, exist_ok=True)  # Create 'logs' folder if it doesn't exist
    log_file_path = os.path.join(log_folder, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file_path,
        filemode='w'  # Overwrite the log file each time
    )

def main(args):
    log_file = "train_log.txt"
    setup_logging(log_file)

    # Load dataset
    df = {}
    files = os.listdir(args.input_dr)
    for f in files:
        file_path = os.path.join(args.input_dr, f)
        if not file_path.endswith(".csv"):
            continue  # Skip non-CSV files
        try:
            df[f] = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue  # Skip empty CSV files
    
    ingest_data.fetch_housing_data()
    housing = df["housing.csv"]

    X_train, X_test, y_train, y_test = ingest_data.prepare_data_for_training(housing)
    
    logging.info("Training Linear Regression model...")
    LR_model = train.train_linear_regression(X_train, y_train)
    logging.info("Training Decision Tree model...")
    DT_model = train.train_decision_tree(X_train, y_train)
    logging.info("Training Random Forest model with random tuning...")
    rand_tune_RF_model = train.rand_tune_random_forest(X_train, y_train)
    logging.info("Training Random Forest model with grid tuning...")
    grid_tune_Tuned_RF_model = train.grid_tune_random_forest(X_train, y_train)

    logging.info("Saving trained models...")
    joblib.dump(LR_model, os.path.join(args.output_dr, "linear_reg_model.pkl"))
    joblib.dump(DT_model, os.path.join(args.output_dr, "decision_tree_model.pkl"))
    joblib.dump(rand_tune_RF_model, os.path.join(args.output_dr, "random_forest_model.pkl"))
    joblib.dump(grid_tune_Tuned_RF_model, os.path.join(args.output_dr, "tuned_random_forest_model.pkl"))
    joblib.dump(grid_tune_Tuned_RF_model.best_estimator_, os.path.join(args.output_dr, "final_model.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on input directory and save them to output directory"
    )
    parser.add_argument("input_dr", type=str, help="Path to the dataset directory")
    parser.add_argument("output_dr", type=str, help="Path to the output directory")
    args = parser.parse_args()

    main(args)
