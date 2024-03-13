# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script

## Running the Code

### Prerequisites

- Python 3.8 or higher installed
- Conda package manager

### Setup Conda Environment

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
conda env create -f env.yml
conda activate mle-dev

Running the Script
Once the Conda environment is activated, you can run the script using the following command:

python script.py

