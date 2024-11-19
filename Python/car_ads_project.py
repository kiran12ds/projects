"""
Final Project: Vehicle Advertising Data Analysis
October 15, 2024
Kiranmayie Bethi
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Data Import and Cleaning
# --------------------------------
# Load the dataset
data = pd.read_csv('car_ads_fp.csv', low_memory = False)
#print("Initial Data Shape: ", data.shape)

# initial structure of the dataset
#print("Initial Data Overview:")
#print(data.head())
#print(data.info())

# Data cleaning steps (handling missing values, data types, etc.)
# Remove any duplicates
data = data.drop_duplicates()
#print("Data Shape after Dropping Duplicates: ", data.shape)

# Cleaning the 'Engin_size' column: Remove non-numeric characters like 'L' and spaces
data['Engin_size'] = data['Engin_size'].str.extract(r'(\d+\.?\d*)')  # Use a raw string for the regex

# Convert the cleaned 'Engin_size' column to float
data['Engin_size'] = pd.to_numeric(data['Engin_size'], errors='coerce')

# Check if any NaN values remain after conversion
#print(data['Engin_size'].isnull().sum())

# Fill missing values with the mode for categorical columns
data['Color'] = data['Color'].fillna(data['Color'].mode()[0])
data['Bodytype'] = data['Bodytype'].fillna(data['Bodytype'].mode()[0])
data['Fuel_type'] = data['Fuel_type'].fillna(data['Fuel_type'].mode()[0])

# For numeric columns, fill with mean or median
data['Engin_size'] = data['Engin_size'].fillna(data['Engin_size'].mean())

# Re-check the shape after filling missing values
#print("Data Shape after Filling Missing Values: ", data.shape)

# Step 2: Filter the Dataset
# --------------------------
# Filter by vehicle models: L200, XC90, Sorento, and Outlander
vehicle_models = ['L200', 'XC90', 'Sorento', 'Outlander']
data = data[data['Genmodel'].isin(vehicle_models)]

# Filter by body type: sport utility vehicle and pickup
body_types = ['SUV', 'Pickup']
data = data[data['Bodytype'].isin(body_types)]

# Filter by fuel type: diesel
data = data[data['Fuel_type'] == 'Diesel']

# Find the six most frequently advertised colors
top_colors = data['Color'].value_counts().nlargest(6).index.tolist()

# Filter by the six most frequently advertised colors
data = data[data['Color'].isin(top_colors)]

# Check the shape after filtering to ensure 2380 observations
#print(top_colors)
#print("Data Shape after Filtering: ", data.shape)

# Subsetting relevant columns for analysis (6 variables)
# Choosing 3 numerical and 3 categorical columns for analysis
columns_needed = ['Genmodel', 'Color', 'Bodytype', 'Runned_Miles', 'Engin_size', 'Price']
data = data[columns_needed]

# Ensure categorical variables are properly encoded
categorical_columns = ['Genmodel', 'Color', 'Bodytype']
data[categorical_columns] = data[categorical_columns].apply(lambda col: col.astype('category'))

# Convert categorical columns to numerical codes for regression
data[categorical_columns] = data[categorical_columns].apply(lambda col: col.cat.codes)

# Step 3: Exploratory Data Analysis (EDA)
# ---------------------------------------
# Generate summary statistics for the dataset
#print("Summary Statistics:")
#print(data.describe())

# Plotting distributions for each feature
sns.pairplot(data)
plt.show()

# Correlation heatmap
# Select only numeric columns for the correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Calculate the correlation matrix
corr = numeric_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 4: Splitting Data into Training and Testing Sets
# ----------------------------------------------------
# Define feature variables (X) and target variable (y)
X = data.drop('Price', axis=1)  # Features (6 variables)
y = data['Price']  # Target variable

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Model Building using Extremely Randomized Trees Regression
# ------------------------------------------------------------------
# Initialize the model
model = ExtraTreesRegressor(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
# ------------------------
# Calculate R2 score and RMSE for both training and testing data
train_pred = model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

test_r2 = r2_score(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the evaluation metrics
#print(f"Training R2: {train_r2:.2f}, Training RMSE: {train_rmse:.2f}")
#print(f"Testing R2: {test_r2:.2f}, Testing RMSE: {test_rmse:.2f}")

# Step 7: Feature Importance Analysis
# -----------------------------------
# Extract feature importances
feature_importances = model.feature_importances_
features = X.columns

# Create a DataFrame to store feature importance values
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Print and plot the feature importances
#print("Feature Importances:")
#print(importance_df)

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance for Predicting Vehicle Prices")
plt.show()

# Step 8: Save the Results and Model (Optional)
# ---------------------------------------------
# Save the trained model using joblib or pickle
import joblib
joblib.dump(model, 'vehicle_price_model.pkl')

# Save cleaned data for future use
data.to_csv('cleaned_car_ads_fp_filtered.csv', index=False)

#print("Script Execution Completed.")
