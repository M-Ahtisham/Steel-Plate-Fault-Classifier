import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from ucimlrepo import fetch_ucirepo

file_path = "data/Faults.csv"

# Checks if dataset file exists
if os.path.exists(file_path):
    print("Loading dataset into memory")
    steel_data = pd.read_csv(file_path)
else:
    print("Dataset not found in /data")
    print("Downloading dataset from https://archive.ics.uci.edu/dataset/198/steel+plates+faults")
    # Fetch dataset from UCI
    # Source of this code: https://archive.ics.uci.edu/dataset/198/steel+plates+faults
    steel_plates_faults = fetch_ucirepo(id=198)
    
    # Extract features and targets
    X = steel_plates_faults.data.features
    y = steel_plates_faults.data.targets

    # Combine into one DataFrame
    steel_data = pd.concat([X, y], axis=1)

    # Ensure the data directory exists
    os.makedirs("data", exist_ok=True)
    steel_data.to_csv(file_path, index=False)

# Continue with analysis
print(steel_data.describe())
print(steel_data.head())
print(steel_data.info())

