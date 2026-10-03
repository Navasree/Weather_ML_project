"""
WeatherGPT - Flask API for Machine Learning Rain Prediction
Exposes /health and /predict endpoints on 127.0.0.1:5001.
"""

import sys
from pathlib import Path
from flask import Flask, jsonify, request

# Ensure both ml/ and the workspace root are in sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load existing weather_model.pkl through ml/predict.py
try:
    from ml.predict import predict_rain
except ImportError:
    from predict import predict_rain

app = Flask(__name__)

# Required input features for prediction
REQUIRED_FIELDS = [
    "temp",
    "RH",
    "windSpeed",
    "gustSpeed",
    "DewPt",
    "windDirection",
    "PAR",
]


@app.after_request
def add_cors_headers(response):
    """Allow cross-origin requests for local development."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to verify ML API status."""
    return jsonify({"status": "ML API is running"}), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """
    POST /predict
    Accepts weather atmospheric parameters as JSON and returns the rain prediction.
    """
    if request.method == "OPTIONS":
        return "", 200

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return (
            jsonify(
                {
                    "error": "Invalid request body. Expected a JSON object with weather parameters."
                }
            ),
            400,
        )

    # Validate presence of all required fields
    missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
    if missing_fields:
        return (
            jsonify(
                {
                    "error": f"Missing required fields: {', '.join(missing_fields)}",
                    "required_fields": REQUIRED_FIELDS,
                }
            ),
            400,
        )

    # Validate that fields are valid numeric values
    parsed_values = {}
    for field in REQUIRED_FIELDS:
        val = data[field]
        try:
            parsed_values[field] = float(val)
        except (ValueError, TypeError):
            return (
                jsonify(
                    {
                        "error": f"Invalid numeric value for field '{field}': received {val!r}"
                    }
                ),
                400,
            )

    # Call predict_rain using the loaded model
    try:
        prediction_result = predict_rain(
            temp=parsed_values["temp"],
            RH=parsed_values["RH"],
            windSpeed=parsed_values["windSpeed"],
            gustSpeed=parsed_values["gustSpeed"],
            DewPt=parsed_values["DewPt"],
            windDirection=parsed_values["windDirection"],
            PAR=parsed_values["PAR"],
        )
        return jsonify(dict(prediction_result)), 200
    except Exception as e:
        return (
            jsonify(
                {
                    "error": "Error during model prediction",
                    "details": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("WeatherGPT ML Flask API starting on http://127.0.0.1:5001")
    print("Available endpoints:")
    print("  GET  /health")
    print("  POST /predict")
    print("=" * 60)
    app.run(host="127.0.0.1", port=5001, debug=False)
