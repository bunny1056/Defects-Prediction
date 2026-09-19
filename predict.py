import joblib
import pandas as pd


MODEL_PATH = "models/random_forest.pkl"
FEATURE_PATH = "models/features.pkl"


model = joblib.load(
    MODEL_PATH
)

features = joblib.load(
    FEATURE_PATH
)


def predict_defect(input_data):

    """
    input_data should be a dictionary
    containing all 16 manufacturing parameters.
    """

    X = pd.DataFrame(
        [input_data],
        columns=features
    )

    prediction = model.predict(X)[0]

    probability = model.predict_proba(
        X
    )[0][1]

    if prediction == 1:

        result = "DEFECT"

    else:

        result = "NO DEFECT"

    return result, probability


if __name__ == "__main__":

    sample = {

        "temperature": 75,
        "pressure": 5.2,
        "humidity": 62,
        "speed": 1200,
        "vibration": 3.1,
        "voltage": 230,
        "current": 8.5,
        "flow_rate": 42,
        "ph": 6.8,
        "concentration": 14,
        "feed_rate": 25,
        "mixing_time": 18,
        "cooling_time": 12,
        "material_density": 1.42,
        "particle_size": 80,
        "machine_age": 4
    }

    result, probability = predict_defect(
        sample
    )

    print(
        f"Prediction: {result}"
    )

    print(
        f"Defect Probability: "
        f"{probability:.2%}"
    )
