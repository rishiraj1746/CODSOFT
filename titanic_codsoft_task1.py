# ==============================================
# CODSOFT - TASK 1: Titanic Survival Prediction
# Author: Rishiraj Verma
# ==============================================

# Step 1: Import Libraries
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the Dataset
print("Downloading Titanic dataset...")
path = kagglehub.dataset_download("yasserh/titanic-dataset")
print("Path to dataset files:", path)

# The dataset is usually saved as 'train.csv' inside the folder
data = pd.read_csv(f"{path}/train.csv")
print("\nDataset Loaded Successfully ✅")
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Data Understanding
print("\nDataset Info:")
print(data.info())

# Step 4: Handle Missing Values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Drop columns that are not useful for prediction
data.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)

# Step 5: Encode Categorical Variables
label = LabelEncoder()
data['Sex'] = label.fit_transform(data['Sex'])
data['Embarked'] = label.fit_transform(data['Embarked'])

# Step 6: Define Features and Target
X = data.drop('Survived', axis=1)
y = data['Survived']

# Step 7: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 9: Model Prediction
y_pred = model.predict(X_test)

# Step 10: Model Evaluation
print("\nModel Evaluation:")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 11: Predict on New Data (Example)
sample_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [1],  # 1 = male
    'Age': [25],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [7.25],
    'Embarked': [0]  # 0 = C
})

prediction = model.predict(sample_data)
print("\nPrediction for new sample:", "Survived ✅" if prediction[0] == 1 else "Did not survive ❌")
