
from flask import Flask, render_template, request, send_from_directory
import os, sqlite3
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static','uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = 'database.db'

# Initialize DB if missing
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  email TEXT,
                  emotion TEXT,
                  message TEXT,
                  image_path TEXT,
                  submitted_at TEXT)""")
    conn.commit()
    conn.close()

init_db()

# Simple heuristic emotion detection using OpenCV Haar cascades.
# This is a lightweight fallback so the app runs without TensorFlow.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def detect_emotion(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "Neutral", "Could not read image."
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return "Neutral", "No face detected clearly. Please try another photo."

    (x,y,w,h) = faces[0]
    face_roi = gray[y:y+h, x:x+w]
    smiles = smile_cascade.detectMultiScale(face_roi, scaleFactor=1.7, minNeighbors=22, minSize=(25,25))
    if len(smiles) > 0:
        return "Happy", "You are smiling. You look happy today!"
    else:
        h2 = int(h*0.6)
        mouth_region = gray[y+h2:y+h, x:x+w]
        if mouth_region.size == 0:
            return "Neutral", "Neutral expression detected."
        mean = np.mean(mouth_region)
        if mean < 80:
            return "Sad", "You are frowning. Why are you sad?"
        else:
            return "Neutral", "You look calm and neutral."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form.get('name','Anonymous')
    email = request.form.get('email','')
    file = request.files.get('image', None)

    if not file:
        return "<h3 style='color:red'>No image uploaded. Go back and upload an image.</h3>"

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f"{timestamp}_{file.filename}"
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    emotion, message = detect_emotion(save_path)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO users (name, email, emotion, message, image_path, submitted_at) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                 (name, email, emotion, message, save_path))
    conn.commit()
    conn.close()

    return f"""<h2>Result</h2>
    <p><strong>Detected Emotion:</strong> {emotion}</p>
    <p><strong>Message:</strong> {message}</p>
    <p><img src='/{save_path}' alt='uploaded image' style='max-width:300px;'></p>
    <p><a href='/'>Try another</a></p>"""

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(os.path.join('static','uploads'), filename)

if __name__ == '__main__':
    app.run(debug=True)
