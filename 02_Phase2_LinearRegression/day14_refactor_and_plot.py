"""
Day 14: Refactor + Visualization

Focus:
- Clean linear regression structure
- Train model
- Plot regression line
- Make it closer to real workflow

No sklearn.
Pure NumPy + Matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, learning_rate=0.01, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m = 0
        self.b = 0

    def fit(self, X, y):
        n = len(X)

        for _ in range(self.epochs):
            y_pred = self.m * X + self.b
            error = y - y_pred

            dm = (-2/n) * np.sum(X * error)
            db = (-2/n) * np.sum(error)

            self.m -= self.learning_rate * dm
            self.b -= self.learning_rate * db

    def predict(self, X):
        return self.m * X + self.b


# Dataset
X = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# Train model
model = LinearRegression()
model.fit(X, y)

print("Trained parameters:")
print("m:", round(model.m, 4))
print("b:", round(model.b, 4))

# Predictions
predictions = model.predict(X)

# Plot
plt.scatter(X, y)
plt.plot(X, predictions)
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression from Scratch")
plt.show()

# Observations:
# - Model wrapped inside a class
# - fit() trains the model
# - predict() makes predictions
# - Visualization confirms line fit
