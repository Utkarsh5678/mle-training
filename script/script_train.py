import argparse
import os
import joblib
import pandas as pd
import logging

from housingpriceprediction import ingest_data
from housingpriceprediction import train

def main(args):
    # Set up logging if enabled
    if args.log_file:
        logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info("Starting model training...")

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
    
    # Train models
    LR_model = train.train_linear_regression(X_train, y_train)
    DT_model = train.train_decision_tree(X_train, y_train)
    rand_tune_RF_model = train.rand_tune_random_forest(X_train, y_train)
    grid_tune_Tuned_RF_model = train.grid_tune_random_forest(X_train, y_train)

    # Create the output directory if it doesn't exist
    output_dir = args.output_dr
    os.makedirs(output_dir, exist_ok=True)  

    # Save trained models
    joblib.dump(LR_model, os.path.join(output_dir, "linear_reg_model.pkl"))
    joblib.dump(DT_model, os.path.join(output_dir, "decision_tree_model.pkl"))
    joblib.dump(rand_tune_RF_model, os.path.join(output_dir, "random_forest_model.pkl"))
    joblib.dump(grid_tune_Tuned_RF_model, os.path.join(output_dir, "tuned_random_forest_model.pkl"))
    joblib.dump(grid_tune_Tuned_RF_model.best_estimator_, os.path.join(output_dir, "final_model.pkl"))

    # Log completion if enabled
    if args.log_file:
        logging.info("Model training completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on input directory and save them to output directory"
    )
    parser.add_argument("input_dr", type=str, help="Path to the dataset directory")
    parser.add_argument("output_dr", type=str, help="Path to the output directory")
    parser.add_argument("--log_file", type=str, help="Path to the log file")
    args = parser.parse_args()

    main(args)
