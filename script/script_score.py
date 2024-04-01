import argparse
import os
import joblib
import pandas as pd
import logging  # Add logging module

from housingpriceprediction import ingest_data
from housingpriceprediction import score as scoring

def setup_logging(output_mode, log_file=None, log_folder="logs"):
    os.makedirs(log_folder, exist_ok=True)  # Create the log folder if it doesn't exist
    
    # Print the log folder path for diagnostic purposes
    print("Log folder path:", os.path.abspath(log_folder))
    
    if output_mode == 'file':
        log_file = os.path.join(log_folder, log_file) if log_file else os.path.join(log_folder, "housing_prediction.log")
    else:
        log_file = None
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a' if output_mode == 'file' else 'w',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Diagnostic print statement
    print("Logging setup completed.")  

def main(args):
    log_folder = "log"  # Assuming "log" is the existing log folder in the directory
    setup_logging(args.output_mode, args.output_file, log_folder)

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
    files = os.listdir(args.input_dr)
    for f in files:
        file_path = os.path.join(args.input_dr, f)
        if not file_path.endswith(".csv"):
            continue  
        try:
            df[f] = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue  
    ingest_data.fetch_housing_data()
    housing = df["housing.csv"]
    X_train, X_test, y_train, y_test = ingest_data.prepare_data_for_training(housing)

    # Score models
    for model_name, model in models.items():
        mae = scoring.score_model_mae(model, X_train, y_train)
        rmse = scoring.score_model_rmse(model, X_train, y_train)
        if args.output_mode == "file":
            logging.info(f"Model: {model_name}, MAE: {mae}, RMSE: {rmse}")
        elif args.output_mode == "print":
            print(f"Model: {model_name}, MAE: {mae}, RMSE: {rmse}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score models using the specified dataset"
    )
    parser.add_argument(
        "input_dr", type=str, help="Path to the dataset directory"
    )  
    parser.add_argument("model_", type=str, help="Path to the model directory")  
    parser.add_argument(
        "output_mode",
        choices=["file", "print"],
        default="print",
        help="Output mode: 'file' to save to a file, 'print' to print to console (default: 'print')",  
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to the output file. Required only when output_mode is 'file'.",
    )
    args = parser.parse_args()

    main(args)