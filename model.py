import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import json

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

def train_model():
    X_train = pd.read_csv('data/pre_processing/X_train.csv')
    y_train = pd.read_csv('data/pre_processing/y_train.csv')

    model = RandomForestClassifier()
    model.fit(X_train, y_train.values.ravel())
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.pkl')

if __name__ == "__main__":
    train_model()

def evaluate_model():
    os.makedirs('result', exist_ok=True)
    
    X_test = pd.read_csv('data//pre_processing/X_test.csv')
    y_test = pd.read_csv('data/pre_processing/y_test.csv').values.ravel() 
    model = joblib.load('models/random_forest_model.pkl')

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')  
    recall = recall_score(y_test, y_pred, average='weighted')  
    precision = precision_score(y_test, y_pred, average='weighted')  
    report = classification_report(y_test, y_pred, output_dict=True)

    with open('result/evaluation_report.json', 'w') as f:
        json.dump(report, f, indent=4)
        
    with open('result/accuracy.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}\n')
        f.write(f'F1 Score: {f1:.2f}\n')
        f.write(f'Recall: {recall:.2f}\n')
        f.write(f'Precision: {precision:.2f}\n')

if __name__ == "__main__":
    evaluate_model()

def deploy_model():
    os.makedirs('models', exist_ok=True)
    model_path = 'models/random_forest_model.pkl'
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        joblib.dump(model, 'models/final_model.pkl')
        print("Model deployed successfully.")
    else:
        print(f"Model file {model_path} does not exist. Please train and save the model first.")

if __name__ == "__main__":
    deploy_model()




