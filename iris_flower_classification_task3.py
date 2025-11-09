# ==============================================
# CODSOFT - TASK 3: Iris Flower Classification
# Author: Rishiraj Verma
# ==============================================

# Step 1: Import Libraries
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load the Dataset
print("Downloading Iris dataset...")
path = kagglehub.dataset_download("arshid/iris-flower-dataset")
print("Path to dataset files:", path)

# Load dataset (usually iris.csv)
data = pd.read_csv(f"{path}/Iris.csv")
print("\nDataset Loaded Successfully ✅")
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Data Understanding
print("\nDataset Info:")
print(data.info())
print("\nUnique Species:", data['Species'].unique())

# Step 4: Data Preprocessing
# Drop the Id column as it doesn't help in classification
data.drop(columns=['Id'], inplace=True)

# Encode species names to numeric labels
label = LabelEncoder()
data['Species'] = label.fit_transform(data['Species'])

# Step 5: Define Features and Target
X = data.drop('Species', axis=1)
y = data['Species']

# Step 6: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Model Training (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Model Prediction
y_pred = model.predict(X_test)

# Step 9: Model Evaluation
print("\nModel Evaluation Results:")
print("Accuracy Score:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label.classes_))

# Step 10: Predict for a New Flower Sample
sample = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})
predicted_species = model.predict(sample)
print("\nPrediction for New Sample:")
print("Predicted Species:", label.inverse_transform(predicted_species)[0])
