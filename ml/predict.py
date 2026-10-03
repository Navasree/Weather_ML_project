"""
WeatherGPT - Rain Prediction Inference Script
Loads trained weather_model.pkl and provides the predict_rain function.
"""

from pathlib import Path
from typing import Any, Dict, Tuple, Union
import joblib
import pandas as pd

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weather_model.pkl"

# Expected feature columns in model training order
FEATURE_COLUMNS = [
    "temp",
    "RH",
    "windSpeed",
    "gustSpeed",
    "DewPt",
    "windDirection",
    "PAR",
]

# Cached model instance
_model = None


def load_model():
    """
    Loads and caches the trained weather model from disk.
    """
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}. "
                f"Please run 'python ml/train_model.py' first to train and generate the model."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class RainPredictionResult(dict):
    """
    Flexible prediction result that acts as a standard dictionary,
    supports attribute access (.prediction, .probability), and can be unpacked as a tuple:
    prediction, probability = predict_rain(...)
    """

    def __init__(
        self,
        prediction: str,
        probability: float,
        rain_probability: float,
        no_rain_probability: float,
    ):
        data = {
            "prediction": prediction,
            "probability": probability,
            "rain_probability": rain_probability,
            "no_rain_probability": no_rain_probability,
        }
        super().__init__(data)
        self.prediction = prediction
        self.probability = probability
        self.rain_probability = rain_probability
        self.no_rain_probability = no_rain_probability

    def __getitem__(self, item: Union[str, int]) -> Any:
        if item == 0:
            return self["prediction"]
        if item == 1:
            return self["probability"]
        return super().__getitem__(item)

    def __iter__(self):
        # Enables unpacking: prediction, probability = predict_rain(...)
        yield self["prediction"]
        yield self["probability"]


def predict_rain(
    temp: float,
    RH: float,
    windSpeed: float,
    gustSpeed: float,
    DewPt: float,
    windDirection: float,
    PAR: float,
) -> RainPredictionResult:
    """
    Predicts whether it will rain given current weather atmospheric features.

    Parameters:
    -----------
    temp : float
        Temperature in Celsius
    RH : float
        Relative Humidity in percentage (0 - 100)
    windSpeed : float
        Wind speed (m/s or knots)
    gustSpeed : float
        Gust speed
    DewPt : float
        Dew point in Celsius
    windDirection : float
        Wind direction in degrees (0 - 360)
    PAR : float
        Photosynthetically Active Radiation

    Returns:
    --------
    RainPredictionResult (dict-like with tuple unpacking support):
        - prediction: 'Rain' or 'No Rain'
        - probability: percentage confidence of the prediction (e.g. 85.5)
        - rain_probability: percentage probability of rain
        - no_rain_probability: percentage probability of no rain
    """
    model = load_model()

    # Create input DataFrame with named feature columns
    input_data = pd.DataFrame(
        [
            {
                "temp": float(temp),
                "RH": float(RH),
                "windSpeed": float(windSpeed),
                "gustSpeed": float(gustSpeed),
                "DewPt": float(DewPt),
                "windDirection": float(windDirection),
                "PAR": float(PAR),
            }
        ],
        columns=FEATURE_COLUMNS,
    )

    # Make prediction and obtain class probabilities
    pred_idx = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    prob_no_rain = round(float(probabilities[0]) * 100, 2)
    prob_rain = round(float(probabilities[1]) * 100, 2)

    # Class 1 = Rain, Class 0 = No Rain
    prediction_label = "Rain" if pred_idx == 1 else "No Rain"
    confidence_percentage = round(float(probabilities[pred_idx]) * 100, 2)

    return RainPredictionResult(
        prediction=prediction_label,
        probability=confidence_percentage,
        rain_probability=prob_rain,
        no_rain_probability=prob_no_rain,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("WeatherGPT: Rain Prediction Test")
    print("=" * 60)

    # Sample Test Case 1: Verified rain event (high RH, low PAR, overcast)
    sample_rain = {
        "temp": 19.72,
        "RH": 87.4,
        "windSpeed": 1.01,
        "gustSpeed": 3.02,
        "DewPt": 17.6,
        "windDirection": 179.7,
        "PAR": 73.7,
    }

    # Sample Test Case 2: Clear, sunny, low humidity conditions
    sample_clear = {
        "temp": 28.0,
        "RH": 35.0,
        "windSpeed": 1.2,
        "gustSpeed": 2.0,
        "DewPt": 10.0,
        "windDirection": 45.0,
        "PAR": 850.0,
    }

    print("\nTest Case 1 (Expected: Rainy conditions):")
    res1 = predict_rain(**sample_rain)
    print(f"  Input: {sample_rain}")
    print(f"  Prediction:      {res1['prediction']}")
    print(f"  Probability:     {res1['probability']}%")
    print(f"  Rain Chance:     {res1['rain_probability']}%")

    print("\nTest Case 2 (Expected: Dry / No Rain conditions):")
    res2 = predict_rain(**sample_clear)
    print(f"  Input: {sample_clear}")
    print(f"  Prediction:      {res2['prediction']}")
    print(f"  Probability:     {res2['probability']}%")
    print(f"  No Rain Chance:  {res2['no_rain_probability']}%")

    # Verify tuple unpacking
    pred, prob = predict_rain(**sample_rain)
    print(f"\nTuple unpacking verified: prediction='{pred}', probability={prob}%")
    print("=" * 60)
