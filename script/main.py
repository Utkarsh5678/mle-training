import os
import subprocess
import mlflow


def run_script(script_path, parent_run_id, run_name):
    env = os.environ.copy()
    env["MLFLOW_RUN_ID"] = parent_run_id
    mlflow.start_run(run_name=run_name)
    subprocess.run(["python", script_path], env=env)
    mlflow.end_run()


def main():
    remote_server_uri = "http://0.0.0.0:5000"
    mlflow.set_tracking_uri(remote_server_uri)
    exp_name = "HousePricePrediction1"
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="Parent Run") as parent_run:
        ingest_data_run = os.path.join(
            os.getcwd(),
            "script",
            "ingest.py",
        )
        train_run = os.path.join(
            os.getcwd(),
            "script",
            "script_train.py",
        )
        score_run = os.path.join(
            os.getcwd(),
            "script",
            "script_score.py",
        )

        with mlflow.start_run(run_name="Ingest Data", nested=True) as ingest_data_child_run:
            print("Running ingest.py")
            run_script(ingest_data_run, ingest_data_child_run.info.run_id, "Ingest Script Run")

        with mlflow.start_run(run_name="Train Model", nested=True) as train_child_run:
            print("Running script_train.py")
            run_script(train_run, train_child_run.info.run_id, "Training Script Run")

        with mlflow.start_run(run_name="Score Model", nested=True) as score_child_run:
            print("Running script_score.py")
            run_script(score_run, score_child_run.info.run_id, "Scoring Script Run")

        print("\nParent Run ID:", parent_run.info.run_id)
        print("Ingest Child Run ID:", ingest_data_child_run.info.run_id)
        print("Train Child Run ID:", train_child_run.info.run_id)
        print("Score Child Run ID:", score_child_run.info.run_id)


if __name__ == "__main__":
    main()