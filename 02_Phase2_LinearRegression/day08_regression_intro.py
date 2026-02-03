"""
Day 8: Introduction to Regression

Focus:
- What regression means in data terms
- Relationship between input and output
- Predicting continuous values
- Setting up data for regression

No sklearn.
No training yet.
Just data + NumPy.
"""

import numpy as np

# Input feature (independent variable)
# Example: size of house in sqft
X = np.array([500, 800, 1200, 1600, 2000])

# Output label (dependent variable)
# Example: price of house
y = np.array([25000, 40000, 65000, 90000, 120000])

print("Input feature (X):", X)
print("Target output (y):", y)

# Understanding relationship
print("\nMean of X:", X.mean())
print("Mean of y:", y.mean())

# Simple manual prediction idea (no model yet)
# If size increases, price also increases
new_house_size = 1400
print("\nNew house size:", new_house_size)

# Rough estimation using proportionality
estimated_price = (new_house_size / X.mean()) * y.mean()
print("Estimated price (rough):", estimated_price)

# Observations:
# - Regression predicts continuous values
# - Input and output have a relationship
# - Models will learn this relationship mathematically
# - This is the problem regression tries to solve
