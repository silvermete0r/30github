# pip install -qU oikan

import torch
import numpy as np
import matplotlib.pyplot as plt

# Import OIKAN components
from oikan.model import OIKAN
from oikan.trainer import train
from oikan.symbolic import extract_symbolic_formula

# Parameters for the simple pendulum
m = 1.0  # mass (kg)
l = 1.0  # length (m)
g = 9.81  # acceleration due to gravity (m/s^2)

# Generate training data
theta = np.linspace(-np.pi, np.pi, 100)
theta_dot = np.linspace(-10, 10, 100)
Theta, Theta_dot = np.meshgrid(theta, theta_dot)

# Calculate Lagrange function L = T - V
T = 0.5 * m * l**2 * Theta_dot**2
V = -m * g * l * np.cos(Theta)
L = T - V

# Flatten data for training
X_train = np.column_stack([Theta.ravel(), Theta_dot.ravel()])
y_train = L.ravel()

# Convert to PyTorch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
y_train_torch = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

# Initialize OIKAN model (2 input features -> 1 output)
model = OIKAN(input_dim=2, output_dim=1, hidden_units=10)

# Train the model
train(model, (X_train_torch, y_train_torch), epochs=500, lr=0.01)

# Extract the symbolic approximation of the Lagrange function
symbolic_formula = extract_symbolic_formula(model, X_train, mode='regression')

# Print the symbolic formula
print("OIKAN Extracted Symbolic Formula:", symbolic_formula)

# Accuracy of the model
with torch.no_grad():
    y_pred = model(X_train_torch).numpy().reshape(100, 100)
    mse = np.mean((y_train - y_pred.ravel())**2)
    print("Mean Squared Error:", mse)

# Visualization of OIKAN Approximation
with torch.no_grad():
    y_pred = model(X_train_torch).numpy().reshape(100, 100)

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Theta, Theta_dot, L, cmap='viridis', alpha=0.8)
ax1.set_title("Original Lagrange Function")

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Theta, Theta_dot, y_pred, cmap='plasma', alpha=0.8)
ax2.set_title("OIKAN Approximation")

plt.show()
