import argparse
import os
import logging
import sys

from housingpriceprediction.ingest_data import (
    fetch_housing_data,
    load_housing_data,
    prepare_data_for_training,
)

def save_csv(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data.to_csv(filename, index=False)

def main(output_folder):
    # Create log file path in the root directory
    log_file = os.path.join(os.getcwd(), "log", "housing_prediction.log")
    
    # Create log directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging to both file and console
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(module)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    try:
        fetch_housing_data()
        housing = load_housing_data()
        X_train, X_test, y_train, y_test = prepare_data_for_training(housing)

        os.makedirs(os.path.join(output_folder, "raw"), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "processed"), exist_ok=True)

        save_csv(housing, os.path.join(output_folder, "raw", "housing.csv"))
        save_csv(X_train, os.path.join(output_folder, "processed", "train_set.csv"))
        save_csv(X_test, os.path.join(output_folder, "processed", "test_set.csv"))
        
        logging.info("Data processing and saving completed.")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and create stratified split datasets.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder.")
    args = parser.parse_args()
    main(args.output_folder)
