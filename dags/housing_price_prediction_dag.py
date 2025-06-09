from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os
import logging
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
import glob
import sqlalchemy
from sqlalchemy import create_engine

sys.path.append('/opt/airflow/src')

from parse_cian import main as parse_cian_main

def extract_flat_id(url):
    """Extract flat ID from Cian URL"""
    return url.split('/')[-2]

def preprocess_data():
    """Preprocess the data"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data preprocessing...")
    try:
        processed_dir = Path("data/processed")
        logger.debug(f"Creating processed directory at {processed_dir.absolute()}")
        processed_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = Path("data/raw")
        logger.debug(f"Looking for raw files in {raw_dir.absolute()}")
        raw_files = glob.glob(str(raw_dir / "*.csv"))
        if not raw_files:
            raise FileNotFoundError("No raw data files found in data/raw/")
        
        latest_file = max(raw_files, key=os.path.getctime)
        logger.info(f"Processing file: {latest_file}")

        logger.debug("Reading CSV file...")
        main_dataframe = pd.read_csv(latest_file)
        logger.debug(f"Read {len(main_dataframe)} rows from CSV")
        
        logger.debug("Extracting flat IDs...")
        main_dataframe['url_id'] = main_dataframe['url'].apply(extract_flat_id)
        df = main_dataframe[['url_id', 'total_meters', 'floor', 'floors_count', 'rooms_count', 'price']].set_index('url_id')
        
        logger.debug("Cleaning data...")
        df = df.sort_index()
        df = df.dropna()
        df = df[df['price'] < 1000000000]
        
        logger.debug("Creating feature columns...")
        df["rooms_1"] = (df["rooms_count"] == 1).astype(int)
        df["rooms_2"] = (df["rooms_count"] == 2).astype(int)
        df["rooms_3"] = (df["rooms_count"] == 3).astype(int)
        df["first_floor"] = (df["floor"] == 1).astype(int)
        df["last_floor"] = (df["floor"] == df["floors_count"]).astype(int)

        df = df[['total_meters', 'floors_count', 'floor', 
                'rooms_1', 'rooms_2', 'rooms_3', 'first_floor', 'last_floor', 'price']]
        
        logger.info("\nДатасет после предобработки:")
        logger.info(df.to_string())
        
        logger.info("\nPreprocessed data statistics:")
        logger.info(f"Number of samples after preprocessing: {len(df)}")
        logger.info(f"Price range after preprocessing: {df['price'].min()} - {df['price'].max()}")
        logger.info(f"Average price by floor after preprocessing:")
        logger.info(df.groupby('floor')['price'].mean().to_string())
        
        train_size = int(len(df) * 0.8)  
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
      
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        
        logger.debug(f"Saving train data to {train_path.absolute()}")
        train_df.to_csv(train_path)
        logger.debug(f"Saving test data to {test_path.absolute()}")
        test_df.to_csv(test_path)
        
        logger.info("Saving data to PostgreSQL...")
        engine = create_engine('postgresql://airflow:airflow@postgres:5432/airflow')
        
        df.to_sql('processed_data', engine, if_exists='replace', index=True)
        logger.info("Data successfully saved to PostgreSQL")
        
        logger.info(f"Train data saved to {train_path}")
        logger.info(f"Test data saved to {test_path}")
        logger.info(f"Number of samples in train: {len(train_df)}")
        logger.info(f"Number of samples in test: {len(test_df)}")
        
        return str(train_path), str(test_path)
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}", exc_info=True)  
        raise 

def train_model(**context):
    """Train the model using preprocessed data"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
   
    train_path, test_path = context['task_instance'].xcom_pull(task_ids='preprocess_data')
    logger.info(f"Using train data from: {train_path}")
    logger.info(f"Using test data from: {test_path}")
  
    train_data = pd.read_csv(train_path, index_col='url_id')
    test_data = pd.read_csv(test_path, index_col='url_id')
 
    X_train = train_data.drop('price', axis=1)
    y_train = train_data['price']
    X_test = test_data.drop('price', axis=1)
    y_test = test_data['price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Model training completed")
    logger.info(f"Train R2 score: {train_score:.4f}")
    logger.info(f"Test R2 score: {test_score:.4f}")
 
    models_dir = Path('/opt/airflow/data/models')
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / 'model.pkl'
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {model_path}")
    
    return str(model_path)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'housing_price_prediction',
    default_args=default_args,
    description='Pipeline for housing price prediction',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

parse_data = PythonOperator(
    task_id='parse_cian_data',
    python_callable=parse_cian_main,
    dag=dag,
)

preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

parse_data >> preprocess >> train 