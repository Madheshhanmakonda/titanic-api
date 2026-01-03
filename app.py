import pandas as pd 
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app=FastAPI()

data=joblib.load("titanic_logistic_pipeline.pkl")

model = data["model"]
threshold = data["threshold"]

class TitanicInput(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
@app.post("/predict") 

def predict_surviver(data:List[TitanicInput]) :
    df=pd.DataFrame([d.dict() for d in data]) 
    prob=model.predict_proba(df)[:,1]
    prediction=(prob>=threshold).astype(int) 
    result=[] 

    for proba,pred in zip(prob,prediction) :        
        result.append({"survival_probability": round(proba, 3),
            "survival_prediction": int(pred),
            "threshold_used": threshold})

    return result