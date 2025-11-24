### End-to-End Customer Telco Churn Prediction
### Purpose

Built an  end-to-end machine-learning solution for predicting customer churn analysis for telecom services. From the data prep and modeling to an API + web UI deployed on locally using FastAPI.

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Repeatable delivery: CI/CD + containers mean every change can be rebuilt, tested, and redeployed in a consistent way.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Perfomred EDA ,data cleaning, feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.
- CI/CD: GitHub Actions automates the creation docker image and pushes it to dockerhub. 
