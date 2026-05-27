import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from flask import Flask, request, jsonify, render_template # Add render_template here
app = Flask(__name__)

model = MobileNetV2(weights='imagenet')
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client['aura_lens_db']
history_collection = db['scans']
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
        
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        label = decode_predictions(predictions, top=1)[0][0]
        # Existing logic
        predictions = model.predict(img_array)
        label = decode_predictions(predictions, top=1)[0][0]

        # ---- INSERT THIS BLOCK HERE ----
        try:
            history_collection.insert_one({
                "object_detected": str(label[1]),
                "confidence": float(label[2]),
                "filename": str(file.filename)
            })
        except Exception as db_err:
            print(f"MongoDB save failed: {db_err}")
        # --------------------------------

        # Existing return block follows right after
        return jsonify({
            "status": "Success",
            "object_detected": label[1],
            "confidence": float(label[2])
        })
        return jsonify({
            "status": "Success",
            "object_detected": label[1],
            "confidence": float(label[2])
        })
    except Exception as e:
        return jsonify({"status": "Error", "message": str(e)}), 500
@app.route('/history', methods=['GET'])
def get_history():
    try:
        # Fetch the 10 most recent scans from MongoDB, excluding the internal database ID
        scans = list(history_collection.find({}, {"_id": 0}).sort("_id", -1).limit(10))
        return jsonify({
            "status": "Success",
            "history": scans
        }), 200
    except Exception as e:
        return jsonify({
            "status": "Error",
            "message": f"Failed to retrieve history: {str(e)}"
        }), 500
if __name__ == '__main__':
    app.run(debug=True)