import pandas as pd 
from src.data.load_data import load_data

def preprocess_data(df:pd.DataFrame,target_col:str = "Churn") -> pd.DataFrame:
    '''
    Basic cleaning for Telco churn. 

    - trim column names 
    - drop id cols 
    - fix Totalcharges to numeric 
    - map target to 0/1
    - simple NA handling
    '''

    df.columns = df.columns.str.strip() # Remove trailing/leading whitespace

    for col in ['customerID','CustomerID','customer_id']:
        if col in df.columns: 
            df = df.drop(columns=[col])
        
    if target_col in df.columns and df[target_col].dtype == 'object':
        df[target_col] = df[target_col].str.strip().map({"No":0,"Yes":1})

    if "TotalCharges" in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')
            
    if "SeniorCitizen" in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
    
    num_cols = df.select_dtypes(include=['number']).columns
    df[num_cols] = df[num_cols].fillna(0)
    
    return df


# if __name__ == '__main__':
#     df = load_data('/home/surya/telco_churn_prediction/data/raw/churn_data.csv')
#     # print(df.head())
#     preprocessed_data = preprocess_data(df)
#     print(preprocessed_data.head())