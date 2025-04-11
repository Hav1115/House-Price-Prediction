import pandas as pd

def load_and_preprocess_data(file_path):
    # Load data from CSV or database
    data = pd.read_csv(file_path)  # Replace with database query if needed

    # Select relevant columns
    data = data[['bedrooms', 'sqft', 'bathrooms', 'price', 'zip_code']]

    # Handle missing values
    data.dropna(inplace=True)

    # Add derived features
    data['price_per_sqft'] = data['price'] / data['sqft']

    # Filter outliers based on Frisco-specific price per sqft range
    data = data[data['price_per_sqft'].between(200, 300)]

    return data

if __name__ == "__main__":
    file_path = "frisco_properties.csv"
    preprocessed_data = load_and_preprocess_data(file_path)
    print(preprocessed_data.head())