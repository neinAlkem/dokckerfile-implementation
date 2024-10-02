import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def download_data():
    # Dataset Iris
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    df = pd.read_csv(url, header=None)
    df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    
    # Save the raw data
    df.to_csv('data/raw/iris.csv', index=False)
    return df

def preprocess_data(df):
    X = df.drop(columns=['species'])
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('data/pre_processing', exist_ok=True)
    
    pd.DataFrame(X_train_scaled, columns=X.columns).to_csv('data/pre_processing/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X.columns).to_csv('data/pre_processing/X_test.csv', index=False)
    y_train.to_csv('data/pre_processing/y_train.csv', index=False)
    y_test.to_csv('data/pre_processing/y_test.csv', index=False)

if __name__ == "__main__":
    df = download_data()
    preprocess_data(df)
