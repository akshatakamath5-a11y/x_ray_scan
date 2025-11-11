import os
import re
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
import joblib
import requests

# Load ML model from pickle
clf = joblib.load('my_model_dsb_pneumonia.pkl')  # Ensure model.pkl is present in the same directory

def preprocess_image(file):
    img = Image.open(file).convert('L').resize((128, 128))
    arr = np.array(img).reshape(-1) / 255.0
    return arr

def predict_pneumonia(image_file):
    features = preprocess_image(image_file)
    features = np.expand_dims(features, axis=0)
    pred = clf.predict(features)[0]
    prob = clf.predict_proba(features)[0].max()
    return int(pred), float(prob)

# Environment variable for deployment (put your key in Render/GitHub config)
GEMINI_API_KEY = os.environ.get('AIzaSyCChIG6PY2gAIBifl0j2_dISSt4oF5HRkc', 'YOUR_FALLBACK_API_KEY')

def get_gemini_advice(diagnosis, confidence):
    prompt = (
        "AI instruction: Respond only with short, clear medical advice for patients based on the X-ray result. "
        "Use two sentences maximum. Avoid special characters like *, #, or - and do not use bullet points. "
        "Write in plain conversational language.\n\n"
        f"Patient X-ray analysis result: {'Pneumonia detected' if diagnosis else 'No pneumonia detected'}, "
        f"confidence: {confidence:.2%}."
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=40)
        data = response.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts:
                return parts[0].get("text", "Please consult a medical professional.")
        return "Please consult a medical professional."
    except Exception as e:
        print("Gemini API error:", e)
        return "Intelligent advice unavailable. Please see a doctor."

def clean_advice(text, max_chars=320):
    # Remove common markdown and special chars, limit length
    text = re.sub(r'[*#_\-]+', '', text)
    text = text.strip()
    # Optionally, ensure two sentences max
    sentences = re.split(r'(?<=[.!?]) +', text)
    text = ' '.join(sentences[:2]).strip()
    # Truncate if somehow too long
    if len(text) > max_chars:
        last_space = text.rfind(' ', 0, max_chars)
        text = text[:last_space].rstrip('.') + '.'
    return text

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    uploaded_file = request.files.get("xray")
    if not uploaded_file:
        return jsonify({"result": "No file uploaded."}), 400
    diagnosis, confidence = predict_pneumonia(uploaded_file)
    return jsonify({"diagnosis": int(diagnosis), "confidence": confidence})

@app.route("/get-advice", methods=["POST"])
def get_advice():
    data = request.json
    diagnosis = data.get("diagnosis")
    confidence = data.get("confidence")
    advice = get_gemini_advice(diagnosis, confidence)
    advice = clean_advice(advice)
    return jsonify({"advice": advice})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render will set PORT; fallback for local dev.
    app.run(debug=False, host="0.0.0.0", port=port)

