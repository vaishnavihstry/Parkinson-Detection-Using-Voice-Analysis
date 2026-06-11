import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# 1️⃣ Load the data
data = pd.read_csv("data.csv")

# 2️⃣ Split into features and target
X = data.drop(columns=["name", "status"])  # drop 'name' and the label column
y = data["status"]

# 3️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Create and fit scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("✅ scaler.pkl created successfully.")

# 5️⃣ Create and train model
model = SVC(probability=True)  # match original model type
model.fit(X_train_scaled, y_train)

# Save model
joblib.dump(model, "model.pkl")
print("✅ model.pkl created successfully.")


