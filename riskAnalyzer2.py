import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

def train_model():
    try:
        data_path = r"C:\Users\oomer\OneDrive\Desktop\projects\NepalEarhquakeDamage2015.csv"
        
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['damage_grade'])
        df['damage_grade'] = df['damage_grade'].str.extract(r'(\d+)').astype(int)

        columns_to_keep = [
            'age_building', 'count_floors_pre_eq', 'plinth_area_sq_ft', 'height_ft_pre_eq',
            'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'damage_grade'
        ]
        df = df[columns_to_keep]

        df.rename(columns={
            'age_building': 'age',
            'count_floors_pre_eq': 'floors',
            'plinth_area_sq_ft': 'area',
            'height_ft_pre_eq': 'height'
        }, inplace=True)

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

        def determine_concrete_type(row):
            if row['roof_type'] in ['Bamboo', 'Thatch', 'Tiled'] and row['ground_floor_type'] in ['Mud', 'Brick']:
                return 'weak'
            elif row['roof_type'] in ['Galvanized', 'Wooden'] and row['ground_floor_type'] in ['Cement', 'Stone']:
                return 'medium'
            elif row['roof_type'] in ['Concrete'] and row['ground_floor_type'] in ['RC', 'Other']:
                return 'strong'
            else:
                return 'unknown'

        df['concrete_type'] = df.apply(determine_concrete_type, axis=1)
        
        df.drop(['foundation_type', 'land_surface_condition', 'roof_type', 'ground_floor_type'], axis=1, inplace=True)
        df = pd.get_dummies(df, columns=['soil_type', 'concrete_type'])

        X = df.drop('damage_grade', axis=1)
        y = df['damage_grade']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, max_depth=None, n_jobs=-1, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs('models', exist_ok=True)
        with open('models/random_forest_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/feature_columns.pkl', 'wb') as f:
            pickle.dump(X.columns.tolist(), f)

        print("Model baþarýyla eðitildi ve kaydedildi!")
        return True

    except Exception as e:
        print(f"Model eðitimi sýrasýnda hata: {e}")
        return False

if _name_ == "_main_":
    train_model()