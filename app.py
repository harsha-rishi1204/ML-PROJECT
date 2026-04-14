"""
=============================================================
  Detection of Fake Online Transactions
  Using Feature Reduction and Margin Maximization
  Backend: Flask + Real Trained RandomForest Model
=============================================================
  Model   : RandomForestClassifier (31 features)
  Features: Time, V1–V28, Amount + 1 padding = 31
  Run     : python app.py
  URL     : http://127.0.0.1:5000
=============================================================

DEPENDENCIES:
    pip install flask flask-cors scikit-learn numpy joblib
"""

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
from flask_cors import CORS

# ─────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
CORS(app)  # Allow requests from any origin (needed for local dev)

# ─────────────────────────────────────────────────────────────
#  LOAD REAL MODEL
#  Place fraud_model.pkl in the same folder as this file.
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")

print("⏳  Loading fraud detection model …")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅  Model loaded: {type(model).__name__}")
    print(f"    Expects {model.n_features_in_} features")
except FileNotFoundError:
    print("❌  fraud_model.pkl not found — place it next to app.py")
    model = None


# ─────────────────────────────────────────────────────────────
#  SERVE FRONTEND
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve index.html from the same directory."""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>index.html not found — place it next to app.py</h1>", 404


# ─────────────────────────────────────────────────────────────
#  PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
      { "Time": float, "Amount": float, "V1"…"V10": float }

    Returns JSON:
      {
        "result":       "Fraud" | "Safe",
        "label":        "Fraud Transaction" | "Genuine Transaction",
        "prediction":   1 | 0,
        "confidence":   float (0–100),
        "fraud_prob":   float,
        "genuine_prob": float
      }
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Check fraud_model.pkl"}), 500

    try:
        data = request.get_json(force=True)

        # ── Validate required fields ───────────────────────────
        required = ["Time", "Amount"] + [f"V{i}" for i in range(1, 11)]
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # ── Build feature vector ───────────────────────────────
        # Model was trained with 31 features in this order:
        #   [Time(1), V1–V10(10 user inputs), V11–V28(18 zeros), Amount(1), extra(1)]
        # Total: 1 + 10 + 18 + 1 + 1 = 31
        # --- New Feature Builder ---

        features = []

        # Time
        features.append(float(data.get("Time", 0)))

        # V1 to V28
        for i in range(1, 29):
            features.append(float(data.get(f"V{i}", 0)))

        # Amount
        features.append(float(data.get("Amount", 0)))

        # Padding
        features.append(0)

        print("Feature Length:", len(features))  # should be 31

        X = np.array(features).reshape(1, -1)

        # ── Predict ────────────────────────────────────────────
        pred  = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]          # [P(class_0), P(class_1)]

        # Handle models where class order might differ
        classes = list(model.classes_)
        if 1 in classes:
            fraud_prob   = round(float(proba[classes.index(1)]) * 100, 2)
            genuine_prob = round(float(proba[classes.index(0)]) * 100, 2)
        else:
            fraud_prob   = round(float(proba[1]) * 100, 2)
            genuine_prob = round(float(proba[0]) * 100, 2)

        confidence = round(max(fraud_prob, genuine_prob), 2)
        label      = "Fraud Transaction" if pred == 1 else "Genuine Transaction"
        result_str = "Fraud" if pred == 1 else "Safe"

        return jsonify({
            "result":       result_str,
            "label":        label,
            "prediction":   pred,
            "confidence":   confidence,
            "fraud_prob":   fraud_prob,
            "genuine_prob": genuine_prob
        })

    except ValueError as ve:
        return jsonify({"error": f"Invalid value: {ve}"}), 400
    except AssertionError as ae:
        return jsonify({"error": str(ae)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  HEALTH CHECK
# ─────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model":  type(model).__name__ if model else "not loaded",
        "features_expected": getattr(model, "n_features_in_", "unknown")
    })


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  FraudShield running at http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
