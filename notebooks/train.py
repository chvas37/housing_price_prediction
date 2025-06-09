import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

def train_model():
    """Train the model and save it"""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv("data/processed/train.csv")

    # scaler = StandardScaler()
    # data['total_meters'] = scaler.fit_transform(data[['total_meters']])

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

    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(y_test - y_pred)):.2f} rubles")
    print(f"Area coefficient: {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    
    model_path = models_dir / "с.pkl"
    
    joblib.dump(model, model_path)
    
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model() 