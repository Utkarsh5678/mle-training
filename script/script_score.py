import argparse
import os
import logging
import joblib
import numpy as np
import pandas as pd
from housingpriceprediction.score import score_model_mae, score_model_rmse
from housingpriceprediction.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Score models.")
    parser.add_argument(
        "--artifacts_path", type=str, default="artifacts",
        help="Path to load model artifacts.")
    parser.add_argument(
        "--processed_path", type=str, default="data/processed",
        help="Path to load processed data.")
    args = parser.parse_args()

    # Load preprocessed data
    X_test = pd.read_csv(f"{args.processed_path}/X_test.csv")
    y_test = pd.read_csv(
        f"{args.processed_path}/y_test.csv"
    ).values.ravel()  # Convert to 1D array

    # Load models
    LR_model = joblib.load(
        f"{args.artifacts_path}/Linear_Regression_Model.pkl"
    )
    DT_model = joblib.load(
        f"{args.artifacts_path}/Decision_Tree_Model.pkl"
    )
    rand_tune_RF_model = joblib.load(
        f"{args.artifacts_path}/Randomized_Search_Model.pkl"
    )
    grid_tune_RF_model = joblib.load(
        f"{args.artifacts_path}/Grid_Search_Model.pkl"
    )

    # Calculate scores
    lr_mae_score = score_model_mae(LR_model, X_test, y_test)
    lr_rmse_score = score_model_rmse(LR_model, X_test, y_test)
    dt_mae_score = score_model_mae(DT_model, X_test, y_test)
    dt_rmse_score = score_model_rmse(DT_model, X_test, y_test)
    final_score = score_model_rmse(grid_tune_RF_model, X_test, y_test)
    rand_cvres = rand_tune_RF_model.cv_results_
    grid_cvres = grid_tune_RF_model.cv_results_


    # Save scores to files
    metrics_path = os.path.join("log")
    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, "Linear Regression Model Score.txt"), "w") as f:
        f.write("Linear Regression MAE score: " + str(lr_mae_score) + "\n")
        f.write("Linear Regression RMSE score: " + str(lr_rmse_score) + "\n")
    with open(os.path.join(metrics_path, "Decision Tree Model Score.txt"), "w") as f:
        f.write("Decision Tree MAE score: " + str(dt_mae_score) + "\n")
        f.write("Decision Tree RMSE score: " + str(dt_rmse_score) + "\n")
    with open(os.path.join(metrics_path, "rand_rf_score.txt"), "w") as f:
        f.write("Random Forest using RandomizedSearchCV model score:\n")
        for mean_score, params in zip(
                rand_cvres["mean_test_score"], rand_cvres["params"]
            ):
            f.write("{} {}\n".format(np.sqrt(-mean_score), params))
    with open(os.path.join(metrics_path, "grid_rf_score.txt"), "w") as f:
        f.write("Random Forest using GridSearchCV model score: \n")
        for mean_score, params in zip(
                grid_cvres["mean_test_score"], grid_cvres["params"]
            ):
            f.write("{} {}\n".format(np.sqrt(-mean_score), params))
    with open(os.path.join(metrics_path, "final_score.txt"), "w") as f:
        f.write("Final Model Score: " + str(final_score) + "\n")

    print("Scores saved to:", metrics_path)

if __name__ == "__main__":
    main()