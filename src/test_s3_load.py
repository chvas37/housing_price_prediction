import boto3
import joblib
import tempfile
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

def test_s3_connection():
    print("Testing S3 connection...")
    print(f"AWS Access Key ID: {os.getenv('aws_access_key_id')[:5]}...")
    print(f"AWS Secret Access Key: {os.getenv('aws_secret_access_key')[:5]}...")
    
    try:
        # Создаем клиент S3
        s3_client = boto3.client(
            's3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=os.getenv('aws_access_key_id'),
            aws_secret_access_key=os.getenv('aws_secret_access_key')
        )
        print("S3 client created successfully")

        # Проверяем доступ к бакету
        bucket_name = 'pabd25'
        model_key = 'ChernovVA/models/catboost_regression_v1.pkl'
        
        print(f"Checking if bucket {bucket_name} exists...")
        s3_client.head_bucket(Bucket=bucket_name)
        print("Bucket exists and is accessible")
        
        print(f"Checking if model file {model_key} exists...")
        s3_client.head_object(Bucket=bucket_name, Key=model_key)
        print("Model file exists")
        
        # Пробуем скачать модель
        print("Attempting to download model...")
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()  # Закрываем файл перед использованием
        
        try:
            s3_client.download_file(bucket_name, model_key, temp_file.name)
            print(f"Model downloaded to temporary file: {temp_file.name}")
            
            # Пробуем загрузить модель
            print("Attempting to load model...")
            model = joblib.load(temp_file.name)
            print("Model loaded successfully!")
            
            return True
            
        finally:
            # Удаляем временный файл в блоке finally
            try:
                os.unlink(temp_file.name)
                print("Temporary file deleted")
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {str(e)}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e)}")
        return False

if __name__ == "__main__":
    print("Starting S3 connection test...")
    success = test_s3_connection()
    print(f"\nTest {'completed successfully' if success else 'failed'}") 