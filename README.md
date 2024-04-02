# Median Housing Value Prediction

This project models the median house value based on a given housing dataset. The housing data can be downloaded [here](https://raw.githubusercontent.com/ageron/handson-ml/master/). The script includes code to download the data.

## Techniques Used

The following techniques have been implemented:

- Linear Regression
- Decision Tree
- Random Forest

## Steps Performed

1. Data Preparation and Cleaning:
   - Download the data using the provided script.
   - Check and impute missing values.

2. Feature Generation and Correlation:
   - Generate features.
   - Check variables for correlation.

3. Sampling Techniques and Data Split:
   - Evaluate multiple sampling techniques.
   - Split the dataset into training and testing sets.

4. Model Evaluation:
   - Implement and evaluate Linear Regression, Decision Tree, and Random Forest models.
   - Use mean squared error as the final metric for evaluation.

## How to Execute the Script

First you need to clone this repository.

```bash
git clone https://github.com/Utkarsh5678/mle-training.git
```

Before running the script, ensure that you have activated the conda environment `mle-dev` where the required packages are installed.
If the `mle-dev` environment does not exist or you don't have the necessary packages installed in that environment, you can create or update your mle-dev environment with all the necessary packages by using the provided env.yml file which is present inside the deploy/conda directory.

```bash
conda env create -f deploy/conda/env.yml
```

Now we can activate the `mle-dev` environment.

```bash
conda env create -f deploy/conda/env.yml
```

After that execute the command to install the HousePricePrediction package

```bash
pip install -e .
```
Once the environment is activated and the required packages are installed, navigate to the project directory.

```bash
cd mle-training
```

Now that we have navigated to the project directory, we can execute the following commands to execute the script:

```bash
python script/ingest.py data    # Executes the script
```

```bash
python script/script_train.py data/raw artifacts/model --log_file training.log -h
python script/script_train.py data/raw artifacts/model --log_file training.log
```

```bash
python script/script_score.py data/raw artifacts/model file --output_file="scoring.log"  -h
python script/script_score.py data/raw artifacts/model file --output_file="scoring.log"

```bash
cd docs
make html
