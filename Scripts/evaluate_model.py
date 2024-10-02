import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import json
import os

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
