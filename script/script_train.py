import argparse
import joblib
import pandas as pd
import mlflow

from housingpriceprediction.train import (
    grid_tune_random_forest,
    rand_tune_random_forest,
    train_decision_tree,
    train_linear_regression,
)


def main():
    parser = argparse.ArgumentParser(description="Train models.")
    parser.add_argument("--processed_path", type=str, default="data/processed", help="Path to load processed data.")
    parser.add_argument("--output_path", type=str, default="artifacts", help="Path to save trained models.")
    args = parser.parse_args()

    # Load preprocessed data
    X_train = pd.read_csv(f"{args.processed_path}/X_train.csv")
    y_train = pd.read_csv(f"{args.processed_path}/y_train.csv").values.ravel()  # Convert to 1D array

    # Train models using the loaded data
    LR_model = train_linear_regression(X_train, y_train)
    DT_model = train_decision_tree(X_train, y_train)
    rand_tune_RF_model = rand_tune_random_forest(X_train, y_train)
    grid_tune_RF_model = grid_tune_random_forest(X_train, y_train)

    # Save models
    joblib.dump(LR_model, f"{args.output_path}/Linear_Regression_Model.pkl")
    joblib.dump(DT_model, f"{args.output_path}/Decision_Tree_Model.pkl")
    joblib.dump(rand_tune_RF_model, f"{args.output_path}/Randomized_Search_Model.pkl")
    joblib.dump(grid_tune_RF_model, f"{args.output_path}/Grid_Search_Model.pkl")
    
    # Log artifacts
    mlflow.log_artifact(args.output_path)

if __name__ == "__main__":
    main()
