import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
# Use the actual file name of your dataset (Housing.csv)
DATA_PATH = BASE_DIR / "data" / "Housing.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    rename_map = {
        "Date": "date",
        "number of bedrooms": "number_of_bedrooms",
        "number of bathrooms": "number_of_bathrooms",
        "living area": "living_area",
        "lot area": "lot_area",
        "number of floors": "number_of_floors",
        "waterfront present": "waterfront_present",
        "number of views": "number_of_views",
        "condition of the house": "condition_of_the_house",
        "grade of the house": "grade_of_the_house",
        "Area of the house(excluding basement)": "area_of_the_house(excluding_basement)",
        "Area of the basement": "area_of_the_basement",
        "Built Year": "built_year",
        "Renovation Year": "renovation_year",
        "Postal Code": "postal_code",
        "Lattitude": "lattitude",
        "Longitude": "longitude",
        "Number of schools nearby": "number_of_schools_nearby",
        "Distance from the airport": "distance_from_the_airport",
        "Price": "price",
    }

    df.rename(
        columns={old: new for old, new in rename_map.items() if old in df.columns},
        inplace=True,
    )

    return df



def preprocess_data(df):
    """
    Preprocess data specifically for your Housing.csv dataset.
    Columns in your dataset:
    ['id', 'date', 'number_of_bedrooms', 'number_of_bathrooms',
     'living_area', 'lot_area', 'number_of_floors', 'waterfront_present',
     'number_of_views', 'condition_of_the_house', 'grade_of_the_house',
     'area_of_the_house(excluding_basement)', 'area_of_the_basement',
     'built_year', 'renovation_year', 'postal_code', 'lattitude',
     'longitude', 'living_area_renov', 'lot_area_renov',
     'number_of_schools_nearby', 'distance_from_the_airport', 'price']
    """

    df = df.copy()

    # Drop columns that are identifiers / not useful for prediction
    for col in ["id", "date"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    target = "price"

    features = [
    "number_of_bedrooms",
    "number_of_bathrooms",
    "living_area",
    "lot_area",
    "number_of_floors",
    "waterfront_present",
    "number_of_views",
    "condition_of_the_house",
    "grade_of_the_house",
    "area_of_the_house(excluding_basement)",
    "area_of_the_basement",
    "built_year",
    "renovation_year",
    "postal_code",
    "lattitude",
    "longitude",
    "living_area_renov",
    "lot_area_renov",
    "number_of_schools_nearby",
    "distance_from_the_airport",
    ]


    # Select X and y
    X = df[features]
    y = df[target]

    # One-hot encode postal_code (acts like a categorical variable)
    X = pd.get_dummies(X, columns=["postal_code"], drop_first=True)

    return X, y


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error
