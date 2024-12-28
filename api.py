from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
import os

app = FastAPI(title="Nepal Earthquake Damage Prediction API")

model = None
feature_columns = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class BuildingData(BaseModel):
    age: int
    floors: int
    soil_type: int
    concrete_type: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 10,
                "floors": 2,
                "soil_type": 2,
                "concrete_type": 1
            }
        }
    }

def load_model():
    global model, feature_columns
    try:
        current_dir = os.path.dirname(os.path.abspath(_file_))
        model_path = os.path.join(current_dir, 'models', 'random_forest_model.pkl')
        features_path = os.path.join(current_dir, 'models', 'feature_columns.pkl')

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(features_path, 'rb') as f:
            feature_columns = pickle.load(f)
        print("Model baþarýyla yüklendi!")
        return True
    except Exception as e:
        print(f"Model yükleme hatasý: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_model()

@app.get("/")
def read_root():
    return {"message": "Earthquake Damage Prediction API"}

@app.post("/predict")
def predict_damage(data: BuildingData):
    global feature_columns
    if feature_columns is None:
        raise HTTPException(status_code=500, detail="Model henüz yüklenmedi")
        
    try:
        input_data = {
            'age': [data.age],
            'floors': [data.floors],
            'soil_type_soft': [1 if data.soil_type == 1 else 0],
            'soil_type_medium': [1 if data.soil_type == 2 else 0],
            'soil_type_hard': [1 if data.soil_type == 3 else 0],
            'concrete_type_weak': [1 if data.concrete_type == 1 else 0],
            'concrete_type_medium': [1 if data.concrete_type == 2 else 0],
            'concrete_type_strong': [1 if data.concrete_type == 3 else 0]
        }

        for col in feature_columns:
            if col not in input_data:
                input_data[col] = [0]

        input_df = pd.DataFrame(input_data, columns=feature_columns)
        prediction = int(model.predict(input_df)[0])

        return {
            "damage_grade": prediction,
            "success": True
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if _name_ == "_main_":
    uvicorn.run(app, host="0.0.0.0", port=8000)
