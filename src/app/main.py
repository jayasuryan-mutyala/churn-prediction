# FASTAPI + GRADIO SERVING APP 
'''
This application provides a complete serving solution for the churn model 
with API access and gradio webapp. 
Architecture:
- FastAPI: High performance REST API 
- Gradio: WebApp interface
- Pydantic: Data validation and automatic API doc 
'''

from fastapi import FastAPI
from pydantic import BaseModel 
import gradio as gr 
from src.serving.inference import predict

app = FastAPI(title="Telco Customer Prediction API",
              description='ML API for predicting the customer churn in telecom industry',
              version='1.0.0')

@app.get('/')
def root():
    '''
    Health check end point for monitoring and load balancer health checks 
    '''
    return {'status':'ok'}

# request data schema 
# Pydantic model for automatic data validation

class CustomerData(BaseModel):
    '''
    This schema defines the exact 18 features for churn prediction
    '''
    gender: str 
    Partner: str 
    Dependents: str 

    PhoneService: str
    MultipleLines: str

    InternetService: str
    OnlineSecurity: str 
    OnlineBackup: str 
    DeviceProtection: str 
    TechSupport: str 
    StreamingSupport: str 
    StreamingTV: str 
    StreamingMovies: str 

    Contact: str 
    PaperlessBilling: str 
    PaymentMethod: str 

    tenure: int 
    MonthlyCharges: float 
    TotalCharges: float

@app.post('/predict')
def get_prediction(data:CustomerData):
    '''
    Main prediction endpoint for churn prediction 
    This endpoint receives validated customer data via Pydantic 
    Calls inference pipeline to transform features and predict
    The prediction is returned JSON format
    '''
    try:
        # Convert pydantic model to dict
        data = data.model_dump()
        result = predict(data)
        return {"prediction":result}
    except Exception as e:
        return {"error":str(e)}
    

    