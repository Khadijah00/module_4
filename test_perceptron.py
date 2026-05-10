import numpy as np
from preprocessing import load_and_prepare_data
from perceptron import Perceptron

# Load data
X_train, X_test, y_train, y_test, class_names = load_and_prepare_data()

# Test ONE perceptron: Setosa classifier
# Setosa = class 0, so y=1 where label is 0
y_binary_setosa = (y_train == 0).astype(int)

print("Training Perceptron for Setosa (step function)...")
print()

p = Perceptron(learning_rate=0.1, epochs=1000, activation='step')
p.train(X_train, y_binary_setosa, verbose=True)

print()
print(f"Final training accuracy: {p.accuracy_history[-1]:.4f}")
print(f"Total epochs run:        {len(p.accuracy_history)}")
print(f"Final weights:           {p.weights.round(4)}")
print(f"Final bias:              {p.bias:.4f}")