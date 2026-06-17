import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from pymongo import MongoClient  # Database Driver
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

app = Flask(__name__)

# Initialize Pre-trained AI Model
model = MobileNetV2(weights='imagenet')

# Configurations
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- STEP 1: INITIALIZE MONGODB CONNECTION ----
try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=2000)
    db = client['aura_lens_db']
    history_collection = db['scans']
    client.server_info()  # Test connection health
except Exception as e:
    print(f"MongoDB connection warning: {e}. Ensure a local instance is running.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Image Matrix Preprocessing
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # AI Execution Layer
        predictions = model.predict(img_array)
        label = decode_predictions(predictions, top=1)[0][0]
        
        object_name = str(label[1]).replace('_', ' ').title()
        confidence_val = float(label[2])

        # ---- STEP 2: PERSIST RESULT TO MONGODB ----
        try:
            history_collection.insert_one({
                "object_detected": object_name,
                "confidence": confidence_val,
                "filename": str(file.filename)
            })
        except Exception as db_err:
            print(f"Database insertion failed: {db_err}")

        return jsonify({
            "status": "Success",
            "object_detected": object_name,
            "confidence": confidence_val
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

# ---- STEP 3: DYNAMIC ACCENT HISTORY ROUTE ----
@app.route('/history', methods=['GET'])
def get_history():
    try:
        # Fetch the last 6 records from MongoDB to fill your UI dashboard cards
        scans = list(history_collection.find({}, {"_id": 0}).sort("_id", -1).limit(6))
        return jsonify({
            "status": "Success",
            "history": scans
        }), 200
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
