"""
train_model.py
Run this ONCE to train the KNN model and save it:
    python train_model.py
Requires dataset.csv in the same directory.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── Load ──────────────────────────────────────────────────────────────────────
dataset = pd.read_csv('dataset.csv')
print(f"Dataset shape: {dataset.shape}")

# ── Encode categoricals ───────────────────────────────────────────────────────
dataset = pd.get_dummies(dataset, columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])

# ── Scale continuous features ─────────────────────────────────────────────────
scaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

# ── Split ─────────────────────────────────────────────────────────────────────
y = dataset['target']
X = dataset.drop(['target'], axis=1)

feature_columns = list(X.columns)
print(f"Features ({len(feature_columns)}): {feature_columns}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# ── Train KNN (k=8 is best per the notebook) ─────────────────────────────────
model = KNeighborsClassifier(n_neighbors=8)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Heart Disease']))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Save artifacts ────────────────────────────────────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\n✓ Saved: model.pkl, scaler.pkl, feature_columns.pkl")
