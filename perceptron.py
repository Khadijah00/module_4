# =============================================================
# perceptron.py
# Implements the Perceptron Learning Rule from scratch.
# Supports multiple activation functions: step, sigmoid, tanh.
# Trains using: w = w + learning_rate * error * input
# =============================================================

import numpy as np


class Perceptron:
    """
    A single binary perceptron classifier.
    Learns to classify one class against all others (used in OvA).

    Attributes:
        learning_rate: how big each weight update step is
        epochs:        how many times to go through training data
        activation:    which activation function to use
        weights:       learned weights for each feature
        bias:          learned bias term
        accuracy_history: accuracy recorded after each epoch
    """

    def __init__(self, learning_rate=0.1, epochs=1000, activation='step'):
        """
        Initializes perceptron with given hyperparameters.

        Args:
            learning_rate: step size for weight updates (default 0.1)
            epochs:        maximum training iterations (default 1000)
            activation:    'step', 'sigmoid', or 'tanh' (default 'step')
        """
        self.learning_rate   = learning_rate
        self.epochs          = epochs
        self.activation      = activation
        self.weights         = None
        self.bias            = None
        self.accuracy_history = []


    # -----------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # Each takes z (weighted sum) and returns the output prediction
    # -----------------------------------------------------------------

    def _step(self, z):
        """
        Step function: returns 1 if z >= 0, else 0.
        Classic perceptron activation. Hard binary decision.
        """
        return np.where(z >= 0, 1, 0)


    def _sigmoid(self, z):
        """
        Sigmoid function: smoothly maps z to range (0, 1).
        Output represents probability/confidence of class 1.
        Formula: 1 / (1 + e^(-z))
        """
        # Clip z to prevent overflow in exp calculation
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def _tanh(self, z):
        """
        Tanh function: smoothly maps z to range (-1, 1).
        Zero centered. Threshold at 0 for classification.
        Formula: (e^z - e^(-z)) / (e^z + e^(-z))
        """
        return np.tanh(z)


    def _activate(self, z):
        """
        Calls the correct activation function based on
        self.activation setting.

        Args:
            z: weighted sum (scalar or array)
        Returns:
            output of chosen activation function
        """
        if self.activation == 'step':
            return self._step(z)
        elif self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def _threshold(self, output):
        """
        Converts continuous activation output to binary prediction.
        Needed for sigmoid and tanh which output non-binary values.

        Step:    already 0 or 1, no change needed
        Sigmoid: output >= 0.5 → 1, else 0
        Tanh:    output >= 0.0 → 1, else 0
        """
        if self.activation == 'step':
            return output
        elif self.activation == 'sigmoid':
            return np.where(output >= 0.5, 1, 0)
        elif self.activation == 'tanh':
            return np.where(output >= 0.0, 1, 0)


    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    def train(self, X_train, y_binary, verbose=False):
        """
        Trains perceptron using the perceptron learning rule:
        w = w + learning_rate * error * input

        Args:
            X_train:  training features, shape (120, 4)
            y_binary: binary labels for this perceptron (0 or 1)
                      e.g. for Setosa perceptron:
                      Setosa=1, Versicolor=0, Virginica=0
            verbose:  if True, prints accuracy every 100 epochs
        """

        num_flowers  = X_train.shape[0]   # 120
        num_features = X_train.shape[1]   # 4

        # --- Initialize weights randomly (small values) ---
        # Small random weights prevent any feature dominating early
        np.random.seed(42)
        self.weights = np.random.uniform(-0.1, 0.1, num_features)
        self.bias    = 0.0

        self.accuracy_history = []

        # --- Training loop ---
        for epoch in range(self.epochs):

            # Shuffle training data each epoch
            # Prevents bias toward order of flowers
            indices = np.random.permutation(num_flowers)
            X_shuffled = X_train[indices]
            y_shuffled = y_binary[indices]

            # --- Go through each flower one by one ---
            for i in range(num_flowers):

                x = X_shuffled[i]    # one flower: 4 features
                y = y_shuffled[i]    # actual label: 0 or 1

                # Step 1: Calculate weighted sum
                z = np.dot(x, self.weights) + self.bias

                # Step 2: Apply activation function
                output = self._activate(z)

                # Step 3: Threshold to get binary prediction
                y_predicted = self._threshold(output)

                # Step 4: Calculate error
                error = y - y_predicted

                # Step 5: Update weights using perceptron rule
                # w = w + learning_rate * error * input
                # If error = 0: no update (correct prediction)
                # If error = +1: weights pushed up
                # If error = -1: weights pushed down
                self.weights += self.learning_rate * error * x
                self.bias    += self.learning_rate * error

            # --- After each epoch: calculate accuracy ---
            epoch_accuracy = self._calculate_accuracy(X_train, y_binary)
            self.accuracy_history.append(epoch_accuracy)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"Accuracy: {epoch_accuracy:.4f}")

            # --- Check convergence ---
            # If accuracy has been 1.0 for last 10 epochs, stop early
            if len(self.accuracy_history) >= 10:
                last_10 = self.accuracy_history[-10:]
                if all(acc == 1.0 for acc in last_10):
                    if verbose:
                        print(f"  Converged at epoch {epoch+1}")
                    break


    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(self, X):
        """
        Predicts binary output for given input flowers.
        Uses frozen weights from training.

        Args:
            X: flower features, shape (n, 4)
        Returns:
            predictions: array of 0s and 1s
            raw_z:       raw weighted sum before activation
                         used for conflict resolution in OvA
        """
        z      = np.dot(X, self.weights) + self.bias
        output = self._activate(z)
        predictions = self._threshold(output)
        return predictions, z


    # -----------------------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------------------

    def _calculate_accuracy(self, X, y_binary):
        """
        Calculates what fraction of predictions are correct.

        Args:
            X:        flower features
            y_binary: true binary labels
        Returns:
            accuracy: float between 0 and 1
        """
        predictions, _ = self.predict(X)
        correct = np.sum(predictions == y_binary)
        accuracy = correct / len(y_binary)
        return accuracy