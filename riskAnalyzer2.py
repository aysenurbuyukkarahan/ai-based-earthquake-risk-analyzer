# -- coding: utf-8 --
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import random

# Dataset path
data_path = r"D:\\archive (1)\\NepalEarhquakeDamage2015.csv"

# Load the dataset
df = pd.read_csv(data_path)

# Remove rows with missing damage_grade
df = df.dropna(subset=['damage_grade'])
df['damage_grade'] = df['damage_grade'].str.extract(r'(\d+)').astype(int)  # Convert to integer

# Keep only relevant columns
columns_to_keep = [
    'age_building', 'count_floors_pre_eq', 'plinth_area_sq_ft', 'height_ft_pre_eq',
    'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'damage_grade'
]
df = df[columns_to_keep]

# Rename columns
df.rename(columns={
    'age_building': 'age',
    'count_floors_pre_eq': 'floors',
    'plinth_area_sq_ft': 'area',
    'height_ft_pre_eq': 'height'
}, inplace=True)

# Combine features for soil type
def determine_soil_type(row):
    if row['land_surface_condition'] == 'Flat' and row['foundation_type'] in ['Mud-Stone', 'Brick-Wood']:
        return 'soft'
    elif row['land_surface_condition'] == 'Moderate' and row['foundation_type'] in ['Cement-Stone', 'Brick-Concrete']:
        return 'medium'
    elif row['land_surface_condition'] == 'Steep' and row['foundation_type'] in ['RC', 'Other']:
        return 'hard'
    else:
        return 'unknown'

df['soil_type'] = df.apply(determine_soil_type, axis=1)

# Combine features for concrete type
def determine_concrete_type(row):
    if row['roof_type'] in ['Bamboo', 'Thatch', 'Tiled'] and row['ground_floor_type'] in ['Mud', 'Brick']:  # Example categories
        return 'weak'
    elif row['roof_type'] in ['Galvanized', 'Wooden'] and row['ground_floor_type'] in ['Cement', 'Stone']:
        return 'medium'
    elif row['roof_type'] in ['Concrete'] and row['ground_floor_type'] in ['RC', 'Other']:
        return 'strong'
    else:
        return 'unknown'

df['concrete_type'] = df.apply(determine_concrete_type, axis=1)

# Drop original categorical columns
df.drop(['foundation_type', 'land_surface_condition', 'roof_type', 'ground_floor_type'], axis=1, inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, columns=['soil_type', 'concrete_type'])

# Features and target
X = df.drop('damage_grade', axis=1)
y = df['damage_grade']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# Save the model and feature names
os.makedirs('models', exist_ok=True)
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

# Make predictions on test set
y_pred = model.predict(X_test)

# Print results
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# User input for predictions
while True:
    print("\nPlease enter the following building details:")
    try:
        age = int(input("Building age: "))
        floors = int(input("Number of floors: "))
        soil_type = int(input("Soil type (1 for soft, 2 for medium, 3 for hard): "))
        concrete_type = int(input("Concrete type (1 for weak, 2 for medium, 3 for strong): "))

        # Assign random values for roof_type and ground_floor_type based on input
        if concrete_type == 1:  # Weak concrete
            roof_type = random.choice(['Bamboo', 'Thatch', 'Tiled'])
            ground_floor_type = random.choice(['Mud', 'Brick'])
        elif concrete_type == 2:  # Medium concrete
            roof_type = random.choice(['Galvanized', 'Wooden'])
            ground_floor_type = random.choice(['Cement', 'Stone'])
        elif concrete_type == 3:  # Strong concrete
            roof_type = 'Concrete'
            ground_floor_type = random.choice(['RC', 'Other'])

        # Assign random values for land_surface_condition and foundation_type based on input
        if soil_type == 1:  # Soft soil
            land_surface_condition = random.choice(['Flat'])
            foundation_type = random.choice(['Mud-Stone', 'Brick-Wood'])
        elif soil_type == 2:  # Medium soil
            land_surface_condition = random.choice(['Moderate'])
            foundation_type = random.choice(['Cement-Stone', 'Brick-Concrete'])
        elif soil_type == 3:  # Hard soil
            land_surface_condition = random.choice(['Steep'])
            foundation_type = random.choice(['RC', 'Other'])

        # Load feature names
        with open('models/feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)

        # Prepare input data
        input_data = {
            'age': [age],
            'floors': [floors],
            'soil_type_soft': [1 if soil_type == 1 else 0],
            'soil_type_medium': [1 if soil_type == 2 else 0],
            'soil_type_hard': [1 if soil_type == 3 else 0],
            'concrete_type_weak': [1 if concrete_type == 1 else 0],
            'concrete_type_medium': [1 if concrete_type == 2 else 0],
            'concrete_type_strong': [1 if concrete_type == 3 else 0]
        }

        # Fill missing columns
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = [0]

        input_df = pd.DataFrame(input_data, columns=feature_columns)

        # Predict
        prediction = model.predict(input_df)[0]
        print(f"Predicted damage grade: {prediction}")

    except Exception as e:
        print(f"Error: {e}")

    continue_prompt = input("Would you like to make another prediction? (y/n): ")
    if continue_prompt.lower() != 'y':
        break
