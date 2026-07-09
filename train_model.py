import pandas as pd
import numpy as np
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier

# Load train dataset
df = pd.read_csv("train.csv")

if 'id' in df.columns:
     df = df.drop('id', axis=1)

X = df.drop("price_range", axis=1)
y = df["price_range"]

feature_names = X.columns.tolist()
pickle.dump(feature_names, open("features.pkl", "wb"))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model initialization & training
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Predictions & Metrics
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)
conf_mat = confusion_matrix(y_test, y_pred).tolist()

print(f"Model Training completed!")
print(f"Test Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model & scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Extract Dataset Statistics for Streamlit defaults
metadata = {
    "accuracy": float(acc),
    "features": {},
    "feature_importances": {}
}

# Calculate feature importances
importances = model.feature_importances_
for name, importance in zip(feature_names, importances):
    metadata["feature_importances"][name] = float(importance)

# Determine defaults and ranges for each feature
binary_cols = ['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi']
for col in feature_names:
    col_min = float(df[col].min())
    col_max = float(df[col].max())
    col_median = float(df[col].median())
    
    if col in binary_cols:
        col_type = "binary"
        col_default = int(df[col].mode()[0])
    else:
        col_type = "numerical"
        # Round default values for clean UX
        if col_median.is_integer():
            col_default = int(col_median)
        else:
            col_default = round(col_median, 2)
            
    metadata["features"][col] = {
        "type": col_type,
        "min": col_min,
        "max": col_max,
        "default": col_default
    }

# Save metadata to JSON
with open("metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("metadata.json generated successfully with defaults and importances!")
