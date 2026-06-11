import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

df = pd.read_csv("data.csv")

# ❌ Remove the name column (string)
df = df.drop(columns=['name'])   # OR df.drop('name', axis=1)

# ✔ Separate features and label
X = df.drop('status', axis=1)
y = df['status']

# ✔ Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = SVC()
model.fit(X_scaled, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training completed!")
