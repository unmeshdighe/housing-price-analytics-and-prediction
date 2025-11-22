import joblib
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "housing_price_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "model_features.pkl"

model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)


sample = {
    "number_of_bedrooms": 3,
    "number_of_bathrooms": 2,
    "living_area": 2000,
    "lot_area": 5000,
    "number_of_floors": 2,
    "waterfront_present": 0,
    "number_of_views": 2,
    "condition_of_the_house": 4,
    "grade_of_the_house": 8,
    "age_of_house": 10,
    "number_of_schools_nearby": 2,
    "distance_from_the_airport": 30,
    "zipcode": 98001,
    "lattitude": 52.90,
    "longitude": -114.50
}

df = pd.DataFrame([sample])

for col in features:
    if col not in df.columns:
        df[col] = 0

df = df[features]

prediction = model.predict(df)[0]

print("Predicted Price:", prediction)
