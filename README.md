
# 🩺 Breast Cancer Prediction API (Flask + PyTorch)

This project is a **Machine Learning deployment** example using **PyTorch** and **Flask**.  
It predicts whether breast cancer is **benign** or **malignant** using the **Breast Cancer Wisconsin dataset** from `sklearn.datasets`.

---

## 📂 Project Structure
```
breast-cancer-api/
│
├── app.py                          # Flask API script
├── breast_cancer_model.pth         # Trained PyTorch model (saved state_dict)
├── README.md                       # Project documentation
└── requirements.txt                # Dependencies
```

---

## 📦 Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

**requirements.txt**
```
flask
torch
scikit-learn
```

---

## 🚀 How to Run

### 1️⃣ Train the Model (if not already trained)
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define model
class BreastCancerModel(nn.Module):
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = BreastCancerModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train loop
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Save model
torch.save(model.state_dict(), "breast_cancer_model.pth")
```

---

### 2️⃣ Run the API
```bash
python app.py
```

---

## 📡 API Endpoints

### **GET** `/`
Check if API is running.
```json
{
    "message": "Breast Cancer Prediction API is running"
}
```

### **GET** `/test`
Run prediction on example values.
```json
{
    "input": [1.0, 2.0, ..., 30.0],
    "probability_benign": 0.6123456,
    "prediction": "benign"
}
```

### **POST** `/predict`
Make a real prediction.  
**Request JSON:**
```json
{
    "features": [1,2,3,4,5,6,7,8,9,10,
                 11,12,13,14,15,16,17,18,19,20,
                 21,22,23,24,25,26,27,28,29,30]
}
```
**Response JSON:**
```json
{
    "probability_benign": 0.845,
    "prediction": "benign"
}
```

---

## 🛠 How It Works
- **PyTorch model** is trained to classify breast cancer cases  
- **Model is saved** as `breast_cancer_model.pth`  
- **Flask API** loads the trained model and serves predictions  
- Users send feature values (30 numbers) to `/predict`  
- Model returns a probability & class label (`benign` / `malignant`)  

---

## 📌 Notes
- You can host this API on **Heroku, Render, or Railway** for free  
- Dataset: `sklearn.datasets.load_breast_cancer()`  
- For browser testing, use `/test` endpoint  
- For API calls, use **Postman** or Python’s `requests` library  
