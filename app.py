from flask import Flask, jsonify, request
from main import getPrediction

app = Flask(__name__)
@app.route("/predictLetter", methods=["POST"])

def predictLetter():
    image = request.files.get("letter")
    prediction = getPrediction(image)
    return jsonify({
        "Prediction": prediction
    }), 200

if __name__ == "__main__":
    app.run(debug=True)