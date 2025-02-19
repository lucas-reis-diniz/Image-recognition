from flask import Flask, request, jsonify
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)
DB_PATH = "images.db"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Criando o banco de dados
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        histogram BLOB)''')
    conn.commit()
    conn.close()

init_db()

def calculate_histogram(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def save_histogram(filename, histogram):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (filename, histogram) VALUES (?, ?)", 
                   (filename, histogram.tobytes()))
    conn.commit()
    conn.close()

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    hist = calculate_histogram(filepath)
    save_histogram(file.filename, hist)
    
    return jsonify({"message": "Image uploaded and histogram stored"}), 200

def compare_histograms(hist1, hist2):
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

@app.route('/recognize', methods=['POST'])
def recognize_image():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    hist_query = calculate_histogram(filepath)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, histogram FROM images")
    matches = []
    for filename, hist_blob in cursor.fetchall():
        hist_db = np.frombuffer(hist_blob, dtype=np.float32)
        similarity = compare_histograms(hist_query, hist_db)
        matches.append((filename, similarity))
    conn.close()
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return jsonify({"matches": matches[:5]})

if __name__ == '__main__':
    app.run(debug=True)
