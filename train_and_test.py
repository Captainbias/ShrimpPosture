import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv('posture_data.csv')

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model accuracy on test set: {accuracy * 100:.2f}%")

# Save trained model
joblib.dump(model, 'posture_model.pkl')
print(" Trained model saved as 'posture_model.pkl'")
