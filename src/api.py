from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash
import os
from dotenv import load_dotenv
import boto3
import pickle
import pandas as pd
from pathlib import Path

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
auth = HTTPBasicAuth()

# Authentication configuration
USERS = {
    os.getenv('API_USER', 'admin'): os.getenv('API_PASSWORD', 'admin')
}

@auth.verify_password
def verify_password(username, password):
    if username in USERS and check_password_hash(USERS[username], password):
        return username

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)

def load_model_from_s3():
    """Load the latest model from S3"""
    bucket_name = os.getenv('S3_BUCKET_NAME')
    model_key = os.getenv('S3_MODEL_KEY', 'models/model.pkl')
    
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=model_key)
        model = pickle.loads(response['Body'].read())
        return model
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        return None

@app.route('/predict', methods=['POST'])
@auth.login_required
def predict():
    try:
        data = request.get_json()
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])
        
        # Load model from S3
        model = load_model_from_s3()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'predicted_price': float(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 