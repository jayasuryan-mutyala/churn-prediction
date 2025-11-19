import os 
import sys 
import time 
import argparse 
import json 
import joblib

import pandas as pd 
import mlflow 
import mlflow.xgboost 
import mlflow.sklearn

from posthog import project_root 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,recall_score,f1_score,
    precision_score,f1_score,roc_auc_score
)

from xgboost import XGBClassifier 

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# Note on argparse it lets you give inputs to 
# python script from command line instead of writing inside the code

def main(args):
    # Main training pipeline function that orchestrates the complete ML workflow
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    # note: dirname returns complete path and '..' means to go up one level 
    mlruns_path = args.mlflow_uri or "sqlite:////home/surya/mlflow.db"
    
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run(run_name='XGB-Churn-Prediction'):
        mlflow.log_param("model","xgboost")
        mlflow.log_param("threshold",args.threshold)
        mlflow.log_param("test_size",args.test_size)

        # Stage 1: Data Loading
        print("Loading data..")
        df = load_data(args.input)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        print("Preprocessing data..")
        df = preprocess_data(df)
        
        # Stage 2: Data Preprocessing 
        processed_path = os.path.join(project_root,'data','processed','churn_processed.csv')
        os.makedirs(os.path.dirname(processed_path),exist_ok=True)
        print(f"Processed dataset saved to {processed_path}| Shape: {df.shape}")

        # Stage 3: Feature engineering 
        print("Building features")
        target = args.target
        if target not in df.columns:
            raise ValueError(f"Target column {target} not found in data")
        
        df_enc = build_features(df,target_col=target)
        
        for c in df_enc.select_dtypes(include=['bool']).columns:
            df_enc[c] = df_enc[c].astype(int)

        print(f"Feature engineering completed:{df_enc.shape[1]} features")
        
        # Save Feature Metadata for Serving Consistency
        artifact_dir = os.path.join(project_root,'artifacts')
        os.makedirs(artifact_dir,exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        with open(os.path.join(artifact_dir,'feature_columns.json'),"w") as f:
            json.dump(feature_cols,f)

        mlflow.log_text("\n".join(feature_cols),artifact_file='feature_columns.txt')

        # These artifacts ensure training and serving use identical transformations
        preprocessing_artifact = {
            "feature_columns":feature_cols,
            "target":target
        }

        joblib.dump(preprocessing_artifact,os.path.join(artifact_dir,"preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifact_dir,"preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # Stage 4: Train test split 
        print("Splitting the data..")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                         test_size=args.test_size,
                                                         stratify=y,
                                                         random_state=42)
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
        
        # Handle the class imbalance 
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"Class imbalance ratio:{scale_pos_weight:.2f} (applied to positive class)")

        # Stage 5: Training XGBoost model
        print(f"Training the XGBoost model")

        model = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight
        )        

        t0 = time.time()
        model.fit(X_train,y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time",train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        # Stage 6: Model evaluation 
        print("Evaluating model performance")
        t1 = time.time()
        proba = model.predict_proba(X_test)[:,1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1

        mlflow.log_metric("pred_time",pred_time)

        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        roc_auc = roc_auc_score(y_test,proba)

        mlflow.log_metric("precision",precision)
        mlflow.log_metric("recall",recall)
        mlflow.log_metric("f1",f1)
        mlflow.log_metric("roc_auc",roc_auc)

        print("Model performance")
        print(f"Precision:{precision:.3f} | Recall:{recall:.3f}")
        print(f"F1 score: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")

        print("Saving model to MLflow..")
        local_feature_file = os.path.join(artifact_dir,"feature_columns.txt")
        
        with open(local_feature_file,'w') as f:
            f.write('\n'.join(feature_cols))
        
        mlflow.xgboost.log_model(model,artifact_path='model')
        mlflow.log_artifact(local_feature_file)
        json_feature_file = os.path.join(artifact_dir, "feature_columns.json")
        if os.path.exists(json_feature_file):
            mlflow.log_artifact(json_feature_file)



        print("Performance Summary:")
        print(f"Training time: {train_time:.2f}s")
        print(f"Inference time: {pred_time:.4f}s")
        print(f"Samples per second: {len(X_test)/pred_time:.0f}")
        
        print(f"\n Detailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost and MLFlow")
    p.add_argument("--input",type=str,required=True,help="path to CSV (e.g., data/raw/Telco-Customer-Churn.csv)")
    p.add_argument("--target",type=str,default="Churn")
    p.add_argument("--threshold",type=float,default=0.35)
    p.add_argument("--test_size",type=float,default=0.2)
    p.add_argument("--experiment",type=str,default="Telco Churn")
    p.add_argument("--mlflow_uri",type=str,default=None,
                     help="override MLflow tracking URI, else uses project_root/mlruns")
    
    args = p.parse_args()
    main(args)