from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Aura Lens API is live and breathing."})

@app.route('/predict', methods=['POST'])
def predict():
    # This is where your AI logic will live soon
    return jsonify({"status": "Success", "message": "Image received for analysis"})

if __name__ == '__main__':
    app.run(debug=True)