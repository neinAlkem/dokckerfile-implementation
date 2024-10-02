import joblib
import os

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
