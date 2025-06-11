import argparse
from flask import Flask, render_template, request, jsonify
from logging.config import dictConfig
from catboost import CatBoostRegressor
from flask_cors import CORS
from functools import wraps
from dotenv import load_dotenv
import os
import boto3
import tempfile
import joblib

# Загружаем переменные окружения из .env файла
load_dotenv()

# Глобальные переменные
model = None
API_TOKEN = os.getenv('API_TOKEN')

dictConfig(
    {
        "version": 1,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "default",
            },
            "file": {
                "class": "logging.FileHandler",
                "filename": "service/flask.log",
                "formatter": "default",
            },
        },
        "root": {"level": "DEBUG", "handlers": ["console", "file"]},
    }
)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Authorization"]}})

def load_model_from_s3():
    try:
        app.logger.info("Starting to load model from S3...")
        s3_client = boto3.client(
            's3',
            endpoint_url='https://storage.yandexcloud.net',
            aws_access_key_id=os.getenv('aws_access_key_id'),
            aws_secret_access_key=os.getenv('aws_secret_access_key')
        )
        app.logger.info("S3 client created successfully")

        bucket_name = 'pabd25'
        model_key = 'ChernovVA/models/catboost_regression_v1.pkl'
        
        app.logger.info(f"Attempting to download model from {bucket_name}/{model_key}")
        
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        
        try:
            s3_client.download_file(bucket_name, model_key, temp_file.name)
            app.logger.info(f"Model downloaded to temporary file: {temp_file.name}")
            
            loaded_model = joblib.load(temp_file.name)
            app.logger.info("Model loaded successfully!")
            
            return loaded_model
            
        finally:
            try:
                os.unlink(temp_file.name)
                app.logger.info("Temporary file deleted")
            except Exception as e:
                app.logger.warning(f"Could not delete temporary file: {str(e)}")
                
    except Exception as e:
        app.logger.error(f"Error loading model from S3: {str(e)}")
        app.logger.error(f"Error type: {type(e)}")
        raise

def init_model(model_path=None):
    global model
    try:
        model = load_model_from_s3()
        app.logger.info("Model loaded successfully from S3")
    except Exception as e:
        app.logger.error(f"Failed to load model from S3: {str(e)}")
        if model_path:
            model = joblib.load(model_path)
            app.logger.info(f"Model loaded from local file: {model_path}")
        else:
            raise Exception("No model available")

init_model()

from flask_httpauth import HTTPTokenAuth
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {API_TOKEN: "user1"} if API_TOKEN else {}

@auth.verify_token
def verify_token(token):
    if not token:
        return None
    if token in tokens:
        return tokens[token]
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/numbers", methods=["POST"])
@auth.login_required
def process_numbers():
    if not request.is_json:
        return jsonify({"status": "error", "data": "Content-Type must be application/json"}), 400

    data = request.get_json()
    app.logger.info(f"Request data: {data}")

    try:
        total_meters = float(data["area"])
        floors_count = int(data["total_floors"])
        floor = int(data["floor"])
        rooms_1 = int(data["rooms"]) == 1
        rooms_2 = int(data["rooms"]) == 2
        rooms_3 = int(data["rooms"]) == 3
        first_floor = int(data["floor"]) == 1
        last_floor = int(data["floor"]) == floors_count
    except ValueError:
        return jsonify({"status": "error", "data": "Ошибка парсинга данных"}), 400

    features = [
        total_meters,
        floors_count,
        floor,
        rooms_1,
        rooms_2,
        rooms_3,
        first_floor,
        last_floor,
    ]

    price = model.predict([features])[0]
    price = int(price)
    return jsonify({"status": "success", "data": price})

if __name__ == "__main__":
    """Parse arguments and run lifecycle steps"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help="Model name", default=None)
    args = parser.parse_args()
    
    if args.model:
        init_model(args.model)
    
    app.run(debug=True)