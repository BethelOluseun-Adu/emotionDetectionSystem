
OYEBANJO_22CD032195 - Face Emotion Detection (Student submission ready)
------------------------------------------------------------
Files included:
- app.py                   : Flask web app (uses OpenCV heuristic to detect happy/neutral/sad)
- model_training.py        : OPTIONAL script to train a neural model (requires TensorFlow and dataset)
- database.db              : SQLite database (will store submissions)
- face_emotionModel.h5     : Placeholder file (not required for app to run)
- requirements.txt         : Python packages to install
- link_web_app.txt         : Paste your deployed URL here after deploying
- templates/index.html     : HTML form (no external CSS)
- static/uploads/          : Where uploaded images are saved

Simple steps to run locally (Windows/mac/Linux):
1. Install Python 3.10+ from python.org (check 'Add to PATH' on Windows).
2. Open a terminal/command prompt and change directory to the project folder:
   cd /path/to/OYEBANJO_22CD032195
3. Install dependencies:
   pip install -r requirements.txt
4. Run the app:
   python app.py
5. Open your browser at http://127.0.0.1:5000 and use the form.

Notes:
- The app uses OpenCV Haar cascades (bundled with opencv-python) to detect faces and smiles.
- If you want a neural network model, you can prepare a dataset and run model_training.py after installing TensorFlow.
- The database (database.db) will record name, email, detected emotion, message, image path, and timestamp.
