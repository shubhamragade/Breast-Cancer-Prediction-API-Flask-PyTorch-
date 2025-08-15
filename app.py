from flask import Flask, request, jsonify
import torch
import torch.nn as nn


# 1. Define model architecture (same as training)

class BreastCancerModel(nn.Module):
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)  # output is probability
        return x

# 2. Load trained model

model = BreastCancerModel()
model.load_state_dict(
    torch.load(
        r"C:\Users\shubh\Desktop\Api-falsk\breast_cancer_model.pth",
        map_location="cpu"
    )
)
model.eval()


# 3. Create Flask app

app = Flask(__name__)

# Health check route
@app.route("/", methods=['GET'])
def home():
    return jsonify({"message": "Breast Cancer Prediction API is running"})

# Prediction route
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()
        features = data.get("features")

        # Validate features
        if not features or len(features) != 30:
            return jsonify({"error": "features must be a list of 30 numbers"}), 400

        # Convert to tensor
        features_tensor = torch.tensor([features], dtype=torch.float32)

        # Make prediction
        with torch.no_grad():
            prob = model(features_tensor).item()  # already sigmoid
            prediction = "benign" if prob > 0.5 else "malignant"

        # Return result
        return jsonify({
            "probability_benign": prob,
            "prediction": prediction
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Test route (GET) - for browser testing
@app.route("/test", methods=['GET'])
def test_prediction():
    # Example input (random values)
    example_features = [float(i) for i in range(1, 31)]
    features_tensor = torch.tensor([example_features], dtype=torch.float32)

    with torch.no_grad():
        prob = model(features_tensor).item()
        prediction = "benign" if prob > 0.5 else "malignant"

    return jsonify({
        "input": example_features,
        "probability_benign": prob,
        "prediction": prediction
    })


# 4. Run the app

if __name__ == "__main__":
    app.run(debug=True)
