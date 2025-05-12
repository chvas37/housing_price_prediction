import pandas as pd
import glob
import os
from pathlib import Path
import numpy as np
import re
from datetime import datetime

def extract_flat_id(url):
    """Extract flat ID from Cian URL"""
    return url.split('/')[-2]

def preprocess_data():
    """Preprocess data from raw CSV files"""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    raw_data_path = "data/raw/"
    file_list = glob.glob(raw_data_path + "*.csv")
    
    if not file_list:
        raise FileNotFoundError("No CSV files found in data/raw/")
    
    latest_file = max(file_list, key=os.path.getctime)
    print(f"Processing file: {latest_file}")
    
    main_dataframe = pd.read_csv(latest_file)
    
    main_dataframe['url_id'] = main_dataframe['url'].apply(extract_flat_id)
    df = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')
    
    df = df.sort_index()
    
    df = df.dropna()
    
    df = df[df['total_meters'] > 0]
    df = df[df['price'] > 0]
    
    df['price_per_meter'] = df['price'] / df['total_meters']
    
    df = df[df['price_per_meter'] <= 100_000_000]

    Q1 = df['price_per_meter'].quantile(0.25)
    Q3 = df['price_per_meter'].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df = df[(df['price_per_meter'] >= lower_bound) & (df['price_per_meter'] <= upper_bound)]
    
    Q1_area = df['total_meters'].quantile(0.25)
    Q3_area = df['total_meters'].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    
    lower_bound_area = Q1_area - 1.5 * IQR_area
    upper_bound_area = Q3_area + 1.5 * IQR_area
    
    df = df[(df['total_meters'] >= lower_bound_area) & (df['total_meters'] <= upper_bound_area)]
    
    df = df.drop('price_per_meter', axis=1)

    test_size = min(10, len(df))
    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_path = processed_dir / f"train_{timestamp}.csv"
    test_path = processed_dir / f"test_{timestamp}.csv"

    train_df.to_csv(train_path)
    test_df.to_csv(test_path)
    
    print(f"Train data saved to {train_path}")
    print(f"Test data saved to {test_path}")
    print(f"Number of samples in train: {len(train_df)}")
    print(f"Number of samples in test: {len(test_df)}")

if __name__ == "__main__":
    preprocess_data() 