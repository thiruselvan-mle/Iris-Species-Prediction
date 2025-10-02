import pandas as pd
import numpy as np
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
    return df

def save_cleaned_data(df):
    import os
    if not os.path.exists("D:\Thiru\ML_Projects\Iris-Species-Prediction\Data\processed"):
        os.makedirs("D:\Thiru\ML_Projects\Iris-Species-Prediction\Data\processed")
    
    df.to_csv("D:\Thiru\ML_Projects\Iris-Species-Prediction\Data\processed\cleaned_iris.csv", index=False)
    print("Cleaned dataset saved to D:\Thiru\ML_Projects\Iris-Species-Prediction\Data\processed\cleaned_iris.csv")
    return df