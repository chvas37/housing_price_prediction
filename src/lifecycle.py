"""This is full life cycle for ml model"""

import argparse
import os
import joblib
import pandas as pd
from pathlib import Path
import sys
import glob
import logging
from datetime import datetime
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import cianparser

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = log_dir / f"lifecycle_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TRAIN_SIZE = 0.8
MODEL_NAME = "linear_regression_v2.pkl"

def parse_cian():
    """Parse data from cian.ru"""
    logger.info("Starting data parsing...")
    try:
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

        moscow_parser = cianparser.CianParser(location="Москва")
        n_rooms = 1
        t = datetime.now().strftime("%Y-%m-%d_%H-%M")
        csv_path = raw_dir / f'{n_rooms}_{t}.csv'
        
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 2,
                "object_type": "secondary"
            })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, encoding='utf-8', index=False)
        logger.info(f"Data saved to {csv_path}")
            
    except Exception as e:
        logger.error(f"Error parsing data: {e}")
        raise

def extract_flat_id(url):
    """Extract flat ID from Cian URL"""
    return url.split('/')[-2]

def preprocess_data():
    """Preprocess the data"""
    logger.info("Starting data preprocessing...")
    try:
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = Path("data/raw")
        raw_files = glob.glob(str(raw_dir / "*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data files found in data/raw/")
        
        latest_file = max(raw_files, key=os.path.getctime)
        logger.info(f"Processing file: {latest_file}")

        main_dataframe = pd.read_csv(latest_file)
        main_dataframe['url_id'] = main_dataframe['url'].apply(extract_flat_id)
        df = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')
        
        df = df.sort_index()
        df = df.dropna()
        df = df[df['price'] < 1000000000]

        train_size = int(len(df) * TRAIN_SIZE)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
      
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        
        train_df.to_csv(train_path)
        test_df.to_csv(test_path)
        
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
        logger.info(f"Number of samples in train: {len(train_df)}")
        logger.info(f"Number of samples in test: {len(test_df)}")
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

def train_model():
    """Train the model"""
    logger.info("Training model...")
    try:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        processed_dir = Path("data/processed")
        train_path = processed_dir / "train.csv"
        
        if not train_path.exists():
            raise FileNotFoundError("Train file not found")
        
        logger.info(f"Using train file: {train_path}")

        data = pd.read_csv(train_path)

        X = data[['total_meters']]
        y = data['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
   
        model = LinearRegression()
        model.fit(X_train, y_train)
       
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
    
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R² Score: {r2:.6f}")
        logger.info(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} rubles")
        logger.info(f"Coefficient: {model.coef_[0]:.2f}")
        logger.info(f"Intercept: {model.intercept_:.2f}")

        model_path = models_dir / MODEL_NAME
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise

def test_model():
    """Test the model"""
    logger.info("Testing model...")
    try:
        model_path = Path("models") / MODEL_NAME
        if not model_path.exists():
            raise FileNotFoundError("Model file not found")
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
  
        processed_dir = Path("data/processed")
        test_path = processed_dir / "test.csv"
        
        if not test_path.exists():
            raise FileNotFoundError("Test file not found")
        
        logger.info(f"Using test file: {test_path}")
 
        test_data = pd.read_csv(test_path)
        X_test = test_data[['total_meters']]
        y_test = test_data['price']
        
        y_pred = model.predict(X_test)
      
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
    
        logger.info("\nTest Results:")
        logger.info(f"Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        logger.info(f"R² Score: {r2:.6f}")
        logger.info(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} rubles")
   
        logger.info("\nSample Predictions:")
        for i in range(min(5, len(y_test))):
            logger.info(f"Actual: {y_test.iloc[i]:.2f}, Predicted: {y_pred[i]:.2f}")
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise

if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    logger.info("Starting housing price prediction pipeline")
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        help="Split data, test relative size, from 0 to 1",
        default=TRAIN_SIZE,
    )
    parser.add_argument("-m", "--model", help="Model name", default=MODEL_NAME)
    args = parser.parse_args()
    
    logger.info(f"Arguments: split={args.split}, model={args.model}")

    try:
        parse_cian()
        preprocess_data()
        train_model()
        test_model()
        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
