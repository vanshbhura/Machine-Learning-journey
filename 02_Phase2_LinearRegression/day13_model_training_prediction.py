"""
Day 13: Model Training and Prediction Workflow

Focus:
- Wrap regression into functions
- Separate training and prediction
- Reusable model structure

No sklearn.
Pure NumPy.
"""

import numpy as np

# Dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])


def train_linear_regression(X, y, learning_rate=0.01, epochs=100):
    m = 0
    b = 0
    n = len(X)

    for i in range(epochs):
        y_pred = m * X + b
        error = y - y_pred

        dm = (-2/n) * np.sum(X * error)
        db = (-2/n) * np.sum(error)

        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b


def predict(X, m, b):
    return m * X + b


# Training
m, b = train_linear_regression(X, y)

print("Trained parameters:")
print("m:", round(m, 4))
print("b:", round(b, 4))

# Predictions
predictions = predict(X, m, b)
print("\nPredictions:", predictions)

# Predict new value
new_x = 6
new_prediction = predict(new_x, m, b)
print("Prediction for x = 6:", new_prediction)

# Observations:
# - Training and prediction are separated
# - Model parameters are reusable
# - Structure now resembles real ML workflow
