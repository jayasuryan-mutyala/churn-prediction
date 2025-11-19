import mlflow 
import pandas as pd
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score,recall_score
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

def train_model(df:pd.DataFrame,target_col:str):
    '''
    Trains an XGBoost model and logs with mlflow
    '''

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    
    
    model = XGBClassifier(n_estimators=300,
                          learning_rate=0.1,
                          max_depth=6,
                          random_state=42,
                          n_jobs=-1,
                          eval_metric='logloss')
    
    
    with mlflow.start_run(run_name='XGBoost clf'):
        # train model
        model.fit(X_train,y_train)
        
        preds = model.predict(X_test)

        acc = accuracy_score(y_test,preds)
        rec = recall_score(y_test,preds)

        mlflow.log_metric('accuracy',acc)
        mlflow.log_metric('recall',rec)
        mlflow.log_param('n_estimators',300)
        mlflow.xgboost.log_model(model,"model")

        # log dataset to mlflow 
        train_dataset = mlflow.data.from_pandas(df,source='training_data')
        mlflow.log_input(train_dataset,context='training')
        print(f"Model trained. Accuracy: {acc:.4f} Recall:{rec:.4f}")


# if __name__ == '__main__':
#     RAW = os.path.join('data','raw','churn_data.csv')
#     df = load_data(RAW)
#     print(df.head())
#     preprocessed_data = preprocess_data(df)
#     print(preprocessed_data.head())

#     cleaned_data = build_features(preprocessed_data)
#     # print(cleaned_data.head())

#     train_model(cleaned_data,target_col='Churn')


