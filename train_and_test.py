import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv('posture_data.csv')

# Split features and labels
X = df.drop('label', axis=1)
y = df['label']

# Train & evaluate using cross-validation
model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5)

# Report accuracy
print(f"\nAverage Accuracy (5-fold CV): {scores.mean() * 100:.2f}%")
print(f"All Fold Scores: {['{:.2f}%'.format(s*100) for s in scores]}")