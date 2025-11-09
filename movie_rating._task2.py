# ==============================================
# CODSOFT - TASK 2: Movie Rating Prediction
# Author: Rishiraj Verma
# ==============================================

# Step 1: Import Libraries
import kagglehub
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load the Dataset
print("Downloading IMDB India Movies dataset...")
path = kagglehub.dataset_download("adrianmcmahon/imdb-india-movies")
print("Path to dataset files:", path)

# Common file name in this dataset
data = pd.read_csv(f"{path}/IMDb India Movies.csv")
print("\nDataset Loaded Successfully ✅")
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Data Understanding
print("\nDataset Info:")
print(data.info())

# Step 4: Data Cleaning
# Drop missing values for simplicity (or fill them if needed)
data.dropna(subset=['Rating'], inplace=True)

# Selecting relevant features
features = ['Genre', 'Director', 'Actors', 'Runtime (Minutes)', 'Year']
target = 'Rating'

# Remove rows with missing feature values
data = data.dropna(subset=features)

# Step 5: Feature Encoding
label = LabelEncoder()
for col in ['Genre', 'Director', 'Actors']:
    data[col] = label.fit_transform(data[col].astype(str))

# Step 6: Define Features and Target
X = data[features]
y = data[target]

# Step 7: Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Model Training (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Model Prediction
y_pred = model.predict(X_test)

# Step 10: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print("Mean Absolute Error (MAE):", round(mae, 2))
print("Mean Squared Error (MSE):", round(mse, 2))
print("Root Mean Squared Error (RMSE):", round(rmse, 2))
print("R² Score:", round(r2, 2))

# Step 11: Predict for New Sample Movie
sample_movie = pd.DataFrame({
    'Genre': [label.transform(['Action'])[0]],  # Example encoding
    'Director': [label.transform(['Rajkumar Hirani'])[0]],
    'Actors': [label.transform(['Aamir Khan'])[0]],
    'Runtime (Minutes)': [150],
    'Year': [2023]
})

predicted_rating = model.predict(sample_movie)
print("\nPredicted Rating for New Movie:", round(predicted_rating[0], 2))
