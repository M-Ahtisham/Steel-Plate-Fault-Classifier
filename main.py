import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

steel_classes = steel_data[steel_data.columns[-7:]]

# Checks if the data is ordered by classes
label_series = steel_classes.idxmax(axis=1)
plt.figure(figsize=(14, 4))
plt.plot(label_series.reset_index(drop=True), marker='.', linestyle='none')
plt.title('Class Distribution Over Row Index')
plt.ylabel('Class')
plt.xlabel('Row Index')
plt.xticks(ticks=range(0, len(label_series), 100))
plt.show()

class_counts = steel_classes.sum()

# Bar plot
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values)

plt.title('Steel Fault Class Distribution')
plt.xlabel('Fault Class')
plt.ylabel('Number of Instances')

# Add numbers on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), 
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# Randomize the dataset
# steel_data = steel_data.sample(frac=1, random_state=12).reset_index(drop=True)

# Separate features and targets
# Assuming the last few columns are fault types (binary targets)
X = steel_data.iloc[:, :-7]  # adjust depending on actual target columns
y = steel_data.iloc[:, -7:]  # 7 target fault types

# Perform train-test split (e.g., 80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Check shapes
print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)
print("Train labels shape:", y_train.shape)
print("Test labels shape:", y_test.shape)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 2: Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Step 3: Predict
y_pred = knn.predict(X_test_scaled)

# Step 4: Convert one-hot to class labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Step 5: Confusion Matrix
conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, normalize='true')
avg_diag = np.trace(conf_matrix) / conf_matrix.shape[0]

# Step 6: Plot
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Relative Confusion Matrix for KNN Classifier")
plt.show()

# Step 7: Output
print(f"Average of diagonal (mean class-wise accuracy): {avg_diag:.4f}")

