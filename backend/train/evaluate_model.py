import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and data
model = joblib.load('models/flood_rf_model.joblib')
data = pd.read_csv('data/synthetic_dataset.csv')

X = data.drop('flood', axis=1)
y = data['flood']

# Predict
y_pred = model.predict(X)

# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy}")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.savefig('models/confusion_matrix.png')
plt.show()

# Feature Importance
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('models/feature_importance.png')
plt.show()

print("Evaluation complete.")
