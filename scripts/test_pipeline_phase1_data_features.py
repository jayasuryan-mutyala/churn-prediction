import sys 
import os 

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

DATA_PATH = os.path.join('data','raw','churn_data.csv')
TARGET_COL = "Churn"


# main pipeline
def main():
    print("Testing Phase 1: Load -> Preprocess -> Build Features")
    
    # Load the data 
    print("\n[1] Loading data...")
    df = load_data(DATA_PATH)
    print(f"DataFrame loaded shape:{df.shape}")
    print(df.head(3))

    # Preprocessing data 
    print("\n[2] Preporcessing data")
    df_clean =  preprocess_data(df,target_col=TARGET_COL)
    print(f"Data after preprocessing shape:{df_clean.shape}")
    print(df_clean.head(3))

    # Build features 
    print(f"\n[3] Building features..")
    df_features = build_features(df_clean,target_col=TARGET_COL)
    print(f"Data after feature engineering shape:{df_features.shape}")
    print(df_features.head(3))

    print("\n Phase 1 pipeline completed successfully")

if __name__ == '__main__':
    main()