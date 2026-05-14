from flask import Flask, request, jsonify
from pymongo import MongoClient
import os

app = Flask(__name__)

# MongoDB setup (using your preferred stack)
client = MongoClient('mongodb://localhost:27017/')
db = client['aura_lens_db']

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "Aura Lens Backend Running", "mode": "Dark"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)