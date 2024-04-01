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
1)First divide the nonstandardcode.py into three parts data_ingestion which imports data and cleans it,methods which will train the cleaned data and has different models under it ,third is score which tells us the score of the predicted models.

2)In the methods file add code to make pickle files in artifacts

3)Build a workflow accordingly that is first set up job,then create repository,setting up miniconda,verify the miniconda environment set up,install tree and create a tree structure before building the package,build the package,upload package artifact and download it ,run main.py and pytest and build the tree again .

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
