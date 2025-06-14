import os
import glob
import logging
from pathlib import Path
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.DEBUG,  
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_flat_id(url):
    """Extract flat ID from Cian URL"""
    return url.split('/')[-2]

def preprocess_data():
    """Preprocess the data"""
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
        df["rooms_1"] = df["rooms_count"] == 1
        df["rooms_2"] = df["rooms_count"] == 2
        df["rooms_3"] = df["rooms_count"] == 3
        df["first_floor"] = df["floor"] == 1
        df["last_floor"] = df["floor"] == df["floors_count"]

        df = df[['total_meters', 'floors_count', 'floor', 
                'rooms_1', 'rooms_2', 'rooms_3', 'first_floor', 'last_floor', 'price']]
        
        print("\nДатасет после предобработки:")
        print(df)
        
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
        
        # Сохраняем данные в PostgreSQL
        logger.info("Saving data to PostgreSQL...")
        engine = create_engine('postgresql://airflow:airflow@postgres:5432/airflow')
        
        # Создаем таблицу processed_data, если она не существует
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

if __name__ == "__main__":
    preprocess_data()

