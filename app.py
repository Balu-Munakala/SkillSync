from flask import Flask, request, jsonify, render_template
import sqlite3
import joblib
import os
import re

app = Flask(__name__)

model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# ---------- Simple ML part ----------

def predict_role(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    proba = max(model.predict_proba(vec)[0])
    return pred, round(proba, 2)

# ---------- Database setup ----------
DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  resume_text TEXT,
                  predicted_role TEXT,
                  confidence REAL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    resume_text = data['resume_text']
    role, confidence = predict_role(resume_text)

    # Save to database
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("INSERT INTO predictions (resume_text, predicted_role, confidence) VALUES (?, ?, ?)",
              (resume_text, role, confidence))
    conn.commit()
    conn.close()

    return jsonify({'role': role, 'confidence': confidence})

@app.route('/dashboard_data')
def dashboard_data():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT predicted_role, COUNT(*) FROM predictions GROUP BY predicted_role")
    data = c.fetchall()
    conn.close()
    return jsonify(data)

if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port=5000)