import argparse
import os
import joblib
import pandas as pd
import logging
import sys
def ingest_logging():
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

def train_logging(log_file=None, log_folder="log"):
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, log_file) if log_file else None
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else None,
            logging.StreamHandler(sys.stdout)
        ],
    )
    logging.info("Starting model training...")