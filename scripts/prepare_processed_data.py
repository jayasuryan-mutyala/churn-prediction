import os 
import sys 

import pandas as pd 

from src.data.preprocess import preprocess_data 
from src.features.build_features import build_features 


RAW = os.path.join('data','raw','churn_data.csv')
OUT = os.path.join('data','processed','telco_churn_processed.csv')

df = pd.read_csv(RAW)
df = preprocess_data(df)

if "Churn" in df.columns and df['Churn'].dtype == 'object':
    df['Churn'] = df['Churn'].str.strip().map({"No":0,"Yes":1}).astype("Int64")

assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

df_processed = build_features(df,target_col='Churn')

os.makedirs(os.path.dirname(OUT),exist_ok=True)
df_processed.to_csv(OUT,index=False)
print(f"Processed dataset to saved {OUT} shape:{df_processed.shape}")
