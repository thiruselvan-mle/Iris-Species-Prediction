import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def ckeck_duplicate(df):
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    df = df.drop_duplicates()
    return df

def drop_cols(df):
    df.drop('Id', axis=1, inplace=True)
    return df

def encode_target(df):
    le=LabelEncoder()
    df['Species']=le.fit_transform(df['Species'])
    df.head()
    return df, le

def save_cleaned_data(df):
    processed_path = r"D:\Thiru\ML_Projects\Iris-Species-Prediction\Data\processed"
    
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    cleaned_file = os.path.join(processed_path, "cleaned_iris.csv")
    df.to_csv(cleaned_file, index=False)
    print(f"Cleaned dataset saved to {cleaned_file}")
    return df  
  

def save_le(le, filename=r"D:\Thiru\ML_Projects\Iris-Species-Prediction\models\le.pkl"):
    joblib.dump(le,filename)
    print(f'Label_Encoded saved to {filename}')
    
def load_le(filename=r"D:\Thiru\ML_Projects\Iris-Species-Prediction\models\le.pkl"):
    le = joblib.load(filename)
    print(f'Label_Encoded loaded from {filename}')
    return le