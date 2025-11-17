import pandas as pd 
import os 

def load_data(file_path: str) -> pd.DataFrame:
    '''
    Loads CSV data into pandas dataframe

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded Dataset.
    '''

    try:
        if os.path.exists:
            return pd.read_csv(file_path) 
        
    except Exception:
        raise FileNotFoundError(f"File not found: {file_path}")
    

# if __name__ == '__main__':
#     RAW = os.path.join('data','raw','churn_data.csv')
#     df = load_data(RAW)
#     print(df.head())