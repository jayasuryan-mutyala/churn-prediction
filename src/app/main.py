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
    

def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        'gender':gender,
        'Partner':Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),              # Ensure integer type
        "MonthlyCharges": float(MonthlyCharges),  # Ensure float type
        "TotalCharges": float(TotalCharges),      # Ensure float type
    }

    result = predict(data)
    return str(result)

demo = gr.Interface(fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        
        # Phone services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        
        # Internet services section (key churn predictors)
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        
        # Contract and billing section (major churn factors)
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        
        # Numeric features (important for churn prediction)
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0, minimum=0, maximum=10000),
    ],

    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.
    
    💡 **Tip**: Month-to-month contracts with fiber optic internet and electronic check payments 
    tend to have higher churn rates.
    """,
    examples=[
        # High churn risk example
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", 
         "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 
         1, 85.0, 85.0],
        # Low churn risk example  
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes",
         "Yes", "No", "No", "Two year", "No", "Credit card (automatic)",
         60, 45.0, 2700.0]
    ],
)

app = gr.mount_gradio_app(
    app,
    demo,
    path='/ui'
)