# =============================================================
# perceptron.py
# Implements the Perceptron Learning Rule from scratch.
# Supports multiple activation functions: step, sigmoid, tanh.
# Update rule: w = w + learning_rate * error * input
#
# FIX: Added class_idx parameter so each OvA binary classifier
#      starts from a unique random seed — consistent with the
#      same fix applied to GradientDescent.
# =============================================================

import numpy as np


class Perceptron:
    """
    Single binary perceptron classifier.
    Learns to separate one class from all others (used inside OvA).

    Training follows the Perceptron Learning Rule:
      For each sample:
        z = w·x + b
        ŷ = threshold(activate(z))
        error = y − ŷ
        w ← w + η * error * x
        b ← b + η * error

    Attributes:
        learning_rate:    step size for weight updates
        epochs:           maximum training iterations
        activation:       'step', 'sigmoid', or 'tanh'
        class_idx:        index of this classifier (0,1,2);
                          used to seed weights uniquely
        weights:          learned weight vector, shape (n_features,)
        bias:             learned bias scalar
        accuracy_history: accuracy recorded after each epoch
    """

    def __init__(self, learning_rate=0.1, epochs=1000,
                 activation='step', class_idx=0):
        """
        Initialises the perceptron with given hyperparameters.

        Args:
            learning_rate: step size for weight updates (default 0.1)
            epochs:        maximum training iterations (default 1000)
            activation:    'step', 'sigmoid', or 'tanh' (default 'step')
            class_idx:     which OvA class this is (default 0).
                           Used so each binary classifier gets unique
                           initial weights.
        """
        self.learning_rate    = learning_rate
        self.epochs           = epochs
        self.activation       = activation
        self.class_idx        = class_idx
        self.weights          = None
        self.bias             = None
        self.accuracy_history = []


    # -----------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # Each maps the weighted sum z to a prediction.
    # -----------------------------------------------------------------

    def _step(self, z):
        """
        Step function: 1 if z ≥ 0, else 0.
        Classic perceptron activation — hard binary decision.
        """
        return np.where(z >= 0, 1, 0)


    def _sigmoid(self, z):
        """
        Sigmoid: smoothly maps z ∈ ℝ → (0, 1).
        Output represents confidence / probability of class 1.
        Formula: σ(z) = 1 / (1 + e^{−z})
        Clipping prevents overflow in e^{−z}.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


    def _tanh(self, z):
        """
        Tanh: smoothly maps z ∈ ℝ → (−1, 1).
        Zero-centred; threshold at 0.0 for classification.
        """
        return np.tanh(z)


    def _activate(self, z):
        """
        Dispatches to the selected activation function.

        Args:
            z: weighted sum, scalar or array
        Returns:
            activation output, same shape as z
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
        Converts continuous activation output to a hard 0/1 prediction.

        Step:    already 0 or 1 — no change
        Sigmoid: ≥ 0.5  → 1,  else 0
        Tanh:    ≥ 0.0  → 1,  else 0
        """
        if self.activation == 'step':
            return output
        elif self.activation == 'sigmoid':
            return np.where(output >= 0.5, 1, 0)
        elif self.activation == 'tanh':
            return np.where(output >= 0.0, 1, 0)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    def train(self, X_train, y_binary, verbose=False):
        """
        Trains the perceptron using the perceptron learning rule.

        Per epoch:
          1. Shuffle training data (prevents order-bias)
          2. For each sample:
               z        = w·x + b
               output   = activate(z)
               ŷ        = threshold(output)
               error    = y − ŷ
               w ← w + η * error * x
               b ← b + η * error
          3. Compute and record epoch accuracy.
          4. Check convergence (perfect accuracy for 10 epochs).

        Args:
            X_train:  feature matrix, shape (120, 4)
            y_binary: binary labels {0,1} for this classifier
            verbose:  print accuracy every 100 epochs if True
        """

        num_samples  = X_train.shape[0]   # 120
        num_features = X_train.shape[1]   # 4

        # --- Unique initial weights per classifier ---
        # Offset seed by class_idx so the 3 OvA classifiers start
        # at different points in weight space.
        np.random.seed(42 + self.class_idx)
        self.weights = np.random.uniform(-0.1, 0.1, num_features)
        self.bias    = 0.0

        self.accuracy_history = []

        # --- Training loop ---
        for epoch in range(self.epochs):

            # Shuffle to prevent bias toward any particular sample order
            indices    = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_binary[indices]

            # --- Online (per-sample) weight updates ---
            for i in range(num_samples):

                x = X_shuffled[i]   # single sample: shape (4,)
                y = y_shuffled[i]   # true label: 0 or 1

                # Forward pass
                z      = np.dot(x, self.weights) + self.bias
                output = self._activate(z)

                # Hard binary prediction for error computation
                y_predicted = self._threshold(output)

                # Error
                error = y - y_predicted

                # Weight update (perceptron rule)
                # error = 0  → no update (prediction was correct)
                # error = +1 → push weights up   (missed a positive)
                # error = −1 → push weights down  (false positive)
                self.weights += self.learning_rate * error * x
                self.bias    += self.learning_rate * error

            # --- Epoch-level accuracy ---
            epoch_accuracy = self._calculate_accuracy(X_train, y_binary)
            self.accuracy_history.append(epoch_accuracy)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"Accuracy: {epoch_accuracy:.4f}")

            # --- Convergence: perfect accuracy for 10 straight epochs ---
            if len(self.accuracy_history) >= 10:
                if all(acc == 1.0 for acc in self.accuracy_history[-10:]):
                    if verbose:
                        print(f"  Converged at epoch {epoch+1}")
                    break


    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(self, X):
        """
        Predicts binary class membership using frozen weights.

        Args:
            X: feature matrix, shape (n, 4)
        Returns:
            predictions: array of 0s and 1s, shape (n,)
            raw_z:       pre-activation weighted sum, shape (n,)
                         used by OvAClassifier for tie-breaking
        """
        z           = np.dot(X, self.weights) + self.bias
        output      = self._activate(z)
        predictions = self._threshold(output)
        return predictions, z


    # -----------------------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------------------

    def _calculate_accuracy(self, X, y_binary):
        """
        Fraction of correct binary predictions on the given data.

        Args:
            X:        feature matrix
            y_binary: true {0,1} labels
        Returns:
            accuracy: float in [0, 1]
        """
        predictions, _ = self.predict(X)
        correct  = np.sum(predictions == y_binary)
        accuracy = correct / len(y_binary)
        return accuracy