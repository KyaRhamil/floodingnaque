import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model_path='models/flood_rf_model.joblib', data_path='data/synthetic_dataset.csv'):
    """Evaluate the trained model and generate metrics."""
    
    # Load model and data
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    model = joblib.load(model_path)
    data = pd.read_csv(data_path)
    
    X = data.drop('flood', axis=1)
    y = data['flood']
    
    # Predict
    y_pred = model.predict(X)
    
    # Accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10, 6))
        feature_importances = pd.Series(model.feature_importances_, index=X.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Evaluation complete.")
    return accuracy

if __name__ == '__main__':
    evaluate_model()