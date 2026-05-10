# =============================================================
# gradient_descent.py
# Implements Gradient Descent Delta Rule from scratch.
# Explicitly minimizes cost function (MSE) using gradients.
# Update rule: w = w - learning_rate * gradient_of_cost
# Uses sigmoid activation and tracks cost per epoch.
# =============================================================

import numpy as np


class GradientDescent:
    """
    Binary classifier using Gradient Descent Delta Rule.
    Minimizes Mean Squared Error cost function.
    Used in OvA strategy same as Perceptron.

    Key difference from Perceptron:
    - Accumulates gradients across ALL flowers
    - Updates weights ONCE per epoch (batch)
    - Explicitly minimizes cost function
    - More stable and systematic than perceptron

    Attributes:
        learning_rate:    step size for weight updates
        epochs:           maximum training iterations
        activation:       activation function to use
        weights:          learned weights
        bias:             learned bias
        cost_history:     cost recorded after each epoch
        accuracy_history: accuracy recorded after each epoch
    """

    def __init__(self, learning_rate=0.1, epochs=1000,
                 activation='sigmoid'):
        """
        Initializes gradient descent classifier.

        Args:
            learning_rate: step size (default 0.1)
            epochs:        max iterations (default 1000)
            activation:    'sigmoid' or 'tanh' (default 'sigmoid')
        """
        self.learning_rate    = learning_rate
        self.epochs           = epochs
        self.activation       = activation
        self.weights          = None
        self.bias             = None
        self.cost_history     = []
        self.accuracy_history = []


    # -----------------------------------------------------------------
    # ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
    # Derivatives needed to calculate gradient of cost function
    # -----------------------------------------------------------------

    def _sigmoid(self, z):
        """
        Sigmoid activation: maps z to (0, 1).
        Formula: 1 / (1 + e^(-z))
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))


    def _sigmoid_derivative(self, output):
        """
        Derivative of sigmoid WITH RESPECT TO z.
        Formula: output * (1 - output)
        Used to calculate how much cost changes
        when we change a weight slightly.

        When output = 0.5 (uncertain): derivative = 0.25 (largest)
        When output = 0.9 (confident): derivative = 0.09 (smaller)
        When output = 0.99 (very confident): derivative = 0.0099 (tiny)
        """
        return output * (1 - output)


    def _tanh(self, z):
        """
        Tanh activation: maps z to (-1, 1).
        Zero centered unlike sigmoid.
        """
        return np.tanh(z)


    def _tanh_derivative(self, output):
        """
        Derivative of tanh WITH RESPECT TO z.
        Formula: 1 - output^2

        When output = 0 (uncertain): derivative = 1 (largest)
        When output = 0.9 (confident): derivative = 0.19
        When output = 0.99: derivative = 0.02 (tiny)
        """
        return 1 - (output ** 2)


    def _activate(self, z):
        """
        Calls correct activation function.
        """
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def _derivative(self, output):
        """
        Calls correct derivative function.
        Matches whichever activation was used.
        """
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(output)
        elif self.activation == 'tanh':
            return self._tanh_derivative(output)


    def _threshold(self, output):
        """
        Converts continuous output to binary prediction.
        Sigmoid: >= 0.5 → 1
        Tanh:    >= 0.0 → 1
        """
        if self.activation == 'sigmoid':
            return np.where(output >= 0.5, 1, 0)
        elif self.activation == 'tanh':
            return np.where(output >= 0.0, 1, 0)


    # -----------------------------------------------------------------
    # COST FUNCTION
    # -----------------------------------------------------------------

    def _calculate_cost(self, X, y_binary):
        """
        Calculates Mean Squared Error cost across all flowers.
        Formula: (1/n) * sum((y - output)^2)

        This is what gradient descent is minimizing.
        Lower cost = better weights = more accurate model.

        Args:
            X:        flower features
            y_binary: true binary labels
        Returns:
            cost: float, average squared error
        """
        z      = np.dot(X, self.weights) + self.bias
        output = self._activate(z)
        cost   = np.mean((y_binary - output) ** 2)
        return cost


    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    def train(self, X_train, y_binary, verbose=False):
        """
        Trains using Batch Gradient Descent.

        Process per epoch:
        1. Forward pass: calculate output for ALL flowers
        2. Calculate error for each flower
        3. Calculate gradient for each flower
        4. AVERAGE gradients across all flowers
        5. Update weights ONCE using averaged gradient

        This is different from perceptron which updates
        after every single flower.

        Args:
            X_train:  training features shape (120, 4)
            y_binary: binary labels for this classifier
            verbose:  print progress every 100 epochs
        """

        num_flowers  = X_train.shape[0]   # 120
        num_features = X_train.shape[1]   # 4

        # --- Initialize weights randomly ---
        np.random.seed(42)
        self.weights = np.random.uniform(-0.1, 0.1, num_features)
        self.bias    = 0.0

        self.cost_history     = []
        self.accuracy_history = []

        # --- Training loop ---
        for epoch in range(self.epochs):

            # ==============================================
            # FORWARD PASS
            # Calculate output for every flower at once
            # numpy handles all 120 flowers simultaneously
            # ==============================================

            # z shape: (120,) one weighted sum per flower
            z = np.dot(X_train, self.weights) + self.bias

            # output shape: (120,) one activation per flower
            output = self._activate(z)

            # ==============================================
            # CALCULATE ERROR
            # ==============================================

            # error shape: (120,) one error per flower
            # positive error: we predicted too low
            # negative error: we predicted too high
            error = y_binary - output

            # ==============================================
            # CALCULATE GRADIENTS
            # gradient = derivative of cost w.r.t. weight
            # tells us: if we increase this weight slightly,
            # does cost go up or down and by how much?
            # ==============================================

            # derivative of activation: shape (120,)
            activation_deriv = self._derivative(output)

            # delta: combines error and activation derivative
            # shape: (120,)
            # This is the core of the delta rule
            delta = error * activation_deriv

            # Gradient for each weight:
            # how much does cost change per unit change in weight
            # X_train.T shape: (4, 120)
            # delta shape:     (120,)
            # result shape:    (4,) one gradient per weight
            # negative because we want to MINIMIZE cost
            grad_weights = -np.dot(X_train.T, delta) / num_flowers
            grad_bias    = -np.mean(delta)

            # ==============================================
            # UPDATE WEIGHTS ONCE (batch update)
            # w = w - learning_rate * gradient
            # Going OPPOSITE to gradient = going downhill
            # ==============================================
            self.weights -= self.learning_rate * grad_weights
            self.bias    -= self.learning_rate * grad_bias

            # ==============================================
            # RECORD COST AND ACCURACY FOR THIS EPOCH
            # ==============================================
            cost     = self._calculate_cost(X_train, y_binary)
            accuracy = self._calculate_accuracy(X_train, y_binary)

            self.cost_history.append(cost)
            self.accuracy_history.append(accuracy)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"Cost: {cost:.6f} | "
                      f"Accuracy: {accuracy:.4f}")

            # --- Check convergence ---
            # Minimum 50 epochs before checking to allow slow learners
            if len(self.cost_history) >= 50:
                cost_change = abs(
                    self.cost_history[-1] - self.cost_history[-2]
                )
                if cost_change < 0.0001:
                    if verbose:
                        print(f"  Converged at epoch {epoch+1} (cost stable)")
                    break

            if len(self.accuracy_history) >= 20:
                if all(acc == 1.0 for acc in self.accuracy_history[-20:]):
                    if verbose:
                        print(f"  Converged at epoch {epoch+1} (perfect accuracy)")
                    break

    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(self, X):
        """
        Predicts binary output for given flowers.
        Uses frozen weights from training.

        Args:
            X: flower features shape (n, 4)
        Returns:
            predictions: binary array (0s and 1s)
            raw_z:       raw weighted sum for conflict resolution
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
        Calculates fraction of correct predictions.

        Args:
            X:        flower features
            y_binary: true binary labels
        Returns:
            accuracy: float 0 to 1
        """
        predictions, _ = self.predict(X)
        correct  = np.sum(predictions == y_binary)
        accuracy = correct / len(y_binary)
        return accuracy