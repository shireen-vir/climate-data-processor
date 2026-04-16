import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    """
    Main function for climate-data-processor.

    This script processes climate data by loading it, splitting it into training and testing sets,
    training a random forest regressor model, and evaluating its performance.
    """
    # Load climate data
    climate_data = pd.read_csv('climate_data.csv')

    # Define features and target
    features = climate_data[['temperature', 'humidity', 'windspeed']]
    target = climate_data['co2_level']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train a random forest regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    # Evaluate the model's performance
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error: {rmse}')

    return rmse

if __name__ == '__main__':
    main()