from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import os

# Load dataset from sklearn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign

# Optional: Save as CSV if you want to inspect it
X['diagnosis'] = y
X.to_csv('data.csv', index=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X.drop(columns=['diagnosis']), y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model to file
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/model.pkl')
print("Model saved to model/model.pkl")
