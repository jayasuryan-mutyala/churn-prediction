import pandas as pd 

from src.data.load_data import load_data 
from src.data.preprocess import preprocess_data

def _map_binary_series(s:pd.Series) -> pd.Series:
    '''
    Apply deterministic binary encoding to 2-category features
    Note that the mappings must be consistent during training and testing
    '''

    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    if valset == {"Yes","No"}:
        return s.map({"No":0,"Yes":1}).astype("Int64")
    
    if valset == {"Male","Female"}:
        return s.map({"Female":0,"Male":1}).astype("Int64")
    
    if  len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]:0,sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")
    
    return s 

def build_features(df:pd.DataFrame,target_col:str = "Churn") -> pd.DataFrame:
    df = df.copy()

    print(f"Starting feature engineering on {df.shape[1]} columns...")
    obj_cols = [c for c in df.select_dtypes(include='object').columns if c != target_col]
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()

    print(f"Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2] 
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f"Binary features: {len(binary_cols)} and Multi-category features: {len(multi_cols)}")

    if binary_cols:
        print(f"Binary features: {binary_cols}")
    
    if multi_cols:
        print(f"Multi categorical features: {multi_cols}")
    

    # applying binary encoding 
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"{c}: {original_dtype} -> binary (0/1)")

    
    # convert boolean cols 
    # XGBoost requires int inputs not boolean
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int: {bool_cols}")
    
    # OHE for multi-catgeory features
    if multi_cols:
        print(f"Applying OHE to {len(multi_cols)} multi-category columns")
        original_shape = df.shape

        df = pd.get_dummies(df,columns=multi_cols,drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"Created {new_features} new features from {len(multi_cols)} categorical columns")

    # Data type cleaning 

    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].fillna(0).astype(int)
    
    print(f"Feature engineering complete: {df.shape[1]} final features")
    return df

# if __name__ == '__main__':
#     RAW = os.path.join('data','raw','churn_data.csv')
#     df = load_data(RAW)
#     print(df.head())    # print(df.head())
#     preprocessed_data = preprocess_data(df)
#     print(preprocessed_data.head())
#     cleaned_data = build_features(preprocessed_data)
#     print(cleaned_data.head())