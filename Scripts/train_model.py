import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def train_model():
    X_train = pd.read_csv('data/pre_processing/X_train.csv')
    y_train = pd.read_csv('data/pre_processing/y_train.csv')

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')

if __name__ == "__main__":
    train_model()
