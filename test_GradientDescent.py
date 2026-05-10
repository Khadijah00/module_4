import numpy as np
from preprocessing import load_and_prepare_data
from gradient_descent import GradientDescent

# Load data
X_train, X_test, y_train, y_test, class_names = load_and_prepare_data()

# Test ONE GD classifier: Setosa
y_binary_setosa = (y_train == 0).astype(int)

print("Training Gradient Descent for Setosa (sigmoid)...")
print()

gd = GradientDescent(learning_rate=0.1, epochs=1000, activation='sigmoid')
gd.train(X_train, y_binary_setosa, verbose=True)

print()
print(f"Final training accuracy: {gd.accuracy_history[-1]:.4f}")
print(f"Final cost:              {gd.cost_history[-1]:.6f}")
print(f"Total epochs run:        {len(gd.cost_history)}")
print(f"Final weights:           {gd.weights.round(4)}")
print(f"Final bias:              {gd.bias:.4f}")