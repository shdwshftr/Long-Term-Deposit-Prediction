from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')
binary_encode = joblib.load('binary_encode.pkl')
label_encode = joblib.load('label_encode.pkl')
ohe_encode = joblib.load('ohe_encode.pkl')
scaler = joblib.load('scaler.pkl')

class Bank(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    duration: float
    campaign: int
    pdays: int
    previous: int
    poutcome: str

@app.get("/")
def read_root():
       return {"message": "Bank Model API"}

@app.post('/predict')

def predict(bank: Bank):
    data = bank.dict()
    df = pd.DataFrame([data])

    df['contact'] = df['contact'].replace(binary_encode['contact'])

    label_columns = ['month', 'day_of_week']
    for col in label_columns:
        df[col] = label_encode[col].transform(df[col])

    ohe_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'poutcome']
    ohe_df = df[ohe_columns]
    ohe_encoded = pd.DataFrame(ohe_encode.transform(ohe_df).toarray(), columns=ohe_encode.get_feature_names_out())

    combined_df = pd.concat([df.drop(columns=ohe_columns), ohe_encoded], axis=1)

    scaled_df = scaler.transform(combined_df)

    prediction = model.predict(scaled_df)
    return {'prediction': prediction[0]}