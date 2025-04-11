from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model(data):
    # Split features and target variable
    X = data[['bedrooms', 'sqft', 'bathrooms', 'price_per_sqft']]
    y = data['price']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
    model.fit(X_train, y_train)

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)

    print(f"RMSE: {rmse}")
    print(f"R²: {r2}")

    return model

if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess_data

    # Load preprocessed data
    file_path = "frisco_properties.csv"
    data = load_and_preprocess_data(file_path)


    trained_model = train_model(data)
    joblib.dump(trained_model, "frisco_price_predictor.pkl")