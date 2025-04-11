import pandas as pd
import sqlalchemy
import joblib


def predict_new_listings(model_file, db_url):
    model = joblib.load(model_file)

    # Can connect to an SQL database
    engine = sqlalchemy.create_engine(db_url)

    # Query new listings from the database
    query = """
        SELECT bedrooms, sqft, bathrooms 
        FROM Properties 
        WHERE city = 'Frisco' AND list_date > CURRENT_DATE - INTERVAL '30 days'
        """

    new_data = pd.read_sql(query, engine)

    # Add feature
    new_data['price_per_sqft'] = 250  # Use average value for Frisco

    # Predict prices
    new_data['predicted_price'] = model.predict(new_data[['bedrooms', 'sqft', 'bathrooms', 'price_per_sqft']])

    return new_data


if __name__ == "__main__":
    db_url = "postgresql://user:password@localhost:5432/real_estate"

    predictions = predict_new_listings("frisco_price_predictor.pkl", db_url)

    print(predictions)