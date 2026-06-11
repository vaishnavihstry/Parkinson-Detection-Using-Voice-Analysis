from flask import Flask, render_template, request, redirect, session
import pickle
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import numpy as np
import os

# --------------------------
# FLASK APP INIT
# --------------------------
app = Flask(__name__)
app.secret_key = "your_secret_key_123"

# --------------------------
# LOAD MODEL
# --------------------------
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = pickle.load(open(model_path, 'rb'))

# --------------------------
# LOAD SCALER
# --------------------------
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
scaler = pickle.load(open(scaler_path, 'rb')) if os.path.exists(scaler_path) else None

# --------------------------
# LOAD DATASET
# --------------------------
csv_path = os.path.join(os.path.dirname(__file__), 'data.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}!")

data = pd.read_csv(csv_path)

# --------------------------
# IDENTIFIER + FEATURES
# --------------------------
identifier_col = data.columns[0]        # first column = person ID
numeric_features = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP",
    "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2"
]

# --------------------------
# INIT DATABASE
# --------------------------
def init_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --------------------------
# REGISTER
# --------------------------
@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']
        confirm  = request.form['confirm_password']

        if password != confirm:
            return render_template("register.html", error="Passwords do not match!")

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()

        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, generate_password_hash(password)))
            conn.commit()
        except:
            conn.close()
            return render_template("register.html", error="Username already exists!")

        conn.close()
        return redirect('/login')

    return render_template("register.html")

# --------------------------
# LOGIN
# --------------------------
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect("users.db")
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE username=?", (username,))
        user = cur.fetchone()
        conn.close()

        if user and check_password_hash(user[0], password):
            session['user'] = username
            return redirect('/')
        else:
            return render_template("login.html", error="Invalid username or password!")

    return render_template("login.html")

# --------------------------
# LOGOUT
# --------------------------
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

# --------------------------
# HOME + PREDICTION
# --------------------------
@app.route('/', methods=['GET','POST'])
def home():
    if 'user' not in session:
        return redirect('/login')

    if request.method == 'POST':
        person_id = request.form['person_id']

        if person_id not in data[identifier_col].values:
            return "Invalid person selected!"

        row = data.loc[data[identifier_col] == person_id, numeric_features]

        # --------------------------
        # FIX: Remove warning by using DataFrame with feature names
        # --------------------------
        input_df = pd.DataFrame([row.values[0]], columns=numeric_features)

        if scaler:
            X = scaler.transform(input_df)
        else:
            X = input_df.values

        prediction = model.predict(X)[0]
        score = model.decision_function(X)[0]
        confidence = round((1 / (1 + np.exp(-score))) * 100, 2)

        result_text = "Healthy" if prediction == 0 else "Parkinson's"

        feature_data = dict(zip(numeric_features, row.values[0]))

        return render_template("result.html",
                               person_id=person_id,
                               prediction=result_text,
                               confidence=confidence,
                               feature_data=feature_data)

    person_ids = data[identifier_col].tolist()
    return render_template("index.html", person_ids=person_ids)

# --------------------------
# MAIN
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
