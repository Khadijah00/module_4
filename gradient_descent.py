# =============================================================
# gradient_descent.py
# Implements Gradient Descent Delta Rule from scratch.
# Explicitly minimizes cost function (MSE) using gradients.
# Update rule: w = w - learning_rate * gradient_of_cost
# Supports sigmoid and tanh activations.
# Tracks cost and accuracy per epoch.
#
# BUG FIXES applied vs original version:
#   BUG 1 — Each classifier now uses a unique random seed
#            (class_idx-based) so the 3 OvA classifiers start
#            with different weights instead of all-identical ones.
#   BUG 2 — tanh now uses ±1 labels internally: the target is
#            converted to {-1, +1} before training so the error
#            signal is mathematically correct for tanh's range.
#            (Original used {0,1} labels with tanh output ∈(-1,1),
#            producing a permanently biased error of up to ±2.)
#   BUG 3 — Convergence threshold tightened from 1e-4 → 1e-6 and
#            minimum-epoch guard raised from 50 → 100 so GD is not
#            stopped prematurely on slowly-improving classifiers.
#   BUG 4 — predict() now converts tanh output using the correct
#            ±1 threshold and remaps back to 0/1 for OvA scoring.
# =============================================================

import numpy as np


class GradientDescent:
    """
    Binary classifier using Gradient Descent Delta Rule.
    Minimizes Mean Squared Error cost function.
    Used inside the OvA strategy, same as Perceptron.

    Key difference from Perceptron:
    - Accumulates gradients across ALL samples per epoch (batch GD)
    - Updates weights ONCE per epoch
    - Explicitly minimizes cost via dCost/dw
    - More stable and systematic than per-sample perceptron updates

    Attributes:
        learning_rate:    step size for weight updates
        epochs:           maximum training iterations
        activation:       'sigmoid' or 'tanh'
        class_idx:        index of this binary classifier (0,1,2)
                          used to ensure unique random seeds
        weights:          learned weight vector  shape (n_features,)
        bias:             learned bias scalar
        cost_history:     MSE cost recorded after each epoch
        accuracy_history: binary accuracy recorded after each epoch
    """

    def __init__(self, learning_rate=0.1, epochs=1000,
                 activation='sigmoid', class_idx=0):
        """
        Initialises the gradient descent binary classifier.

        Args:
            learning_rate: step size (default 0.1)
            epochs:        max training iterations (default 1000)
            activation:    'sigmoid' or 'tanh' (default 'sigmoid')
            class_idx:     which OvA class this is (default 0).
                           Passed by OvAClassifier so each binary
                           classifier starts from different weights.
        """
        self.learning_rate    = learning_rate
        self.epochs           = epochs
        self.activation       = activation
        self.class_idx        = class_idx   # FIX BUG 1
        self.weights          = None
        self.bias             = None
        self.cost_history     = []
        self.accuracy_history = []


    # -----------------------------------------------------------------
    # ACTIVATION FUNCTIONS AND THEIR DERIVATIVES
    # Derivatives needed to backpropagate through the cost function.
    # -----------------------------------------------------------------

    def _sigmoid(self, z):
        """
        Sigmoid activation: maps z ∈ ℝ  →  output ∈ (0, 1).
        Formula: σ(z) = 1 / (1 + e^{-z})
        Numerically stable via clipping before exp.
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))


    def _sigmoid_derivative(self, output):
        """
        Derivative of σ with respect to z, expressed in terms of
        the already-computed output: dσ/dz = output * (1 − output).

        Magnitude intuition
          output = 0.50 → deriv = 0.2500  (maximum, most uncertain)
          output = 0.90 → deriv = 0.0900
          output = 0.99 → deriv = 0.0099  (near-saturated, tiny grad)
        """
        return output * (1.0 - output)


    def _tanh(self, z):
        """
        Tanh activation: maps z ∈ ℝ  →  output ∈ (−1, 1).
        Zero-centred, so gradients are less biased than sigmoid.
        """
        return np.tanh(z)


    def _tanh_derivative(self, output):
        """
        Derivative of tanh with respect to z: d(tanh)/dz = 1 − output².

        Magnitude intuition
          output = 0.00 → deriv = 1.00  (maximum, most uncertain)
          output = 0.90 → deriv = 0.19
          output = 0.99 → deriv = 0.02  (near-saturated, tiny grad)
        """
        return 1.0 - (output ** 2)


    def _activate(self, z):
        """
        Dispatches to the chosen activation function.

        Args:
            z: pre-activation weighted sum, shape (n,) or scalar
        Returns:
            activated output, same shape as z
        """
        if self.activation == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation == 'tanh':
            return self._tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def _derivative(self, output):
        """
        Dispatches to the derivative of the chosen activation.

        Args:
            output: post-activation value (result of _activate)
        Returns:
            derivative value, same shape as output
        """
        if self.activation == 'sigmoid':
            return self._sigmoid_derivative(output)
        elif self.activation == 'tanh':
            return self._tanh_derivative(output)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    def _threshold(self, output):
        """
        Converts continuous activation output to a hard binary
        prediction (0 or 1) for accuracy / OvA scoring.

        Sigmoid: output ≥ 0.5  →  1,  else 0
        Tanh:    output ≥ 0.0  →  1,  else 0
              (tanh output is ±1 internally but threshold maps back
               to the 0/1 space used everywhere else in the project)
        """
        if self.activation == 'sigmoid':
            return np.where(output >= 0.5, 1, 0)
        elif self.activation == 'tanh':
            return np.where(output >= 0.0, 1, 0)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


    # -----------------------------------------------------------------
    # LABEL CONVERSION FOR TANH
    # FIX BUG 2: tanh output lives in (−1, +1). Using {0,1} targets
    # creates a biased error (max error = 2 on the wrong side vs 1 for
    # sigmoid). Convert to {−1, +1} before training so error = target −
    # output is always in [−2, +2] and symmetric around zero.
    # -----------------------------------------------------------------

    def _convert_labels(self, y_binary):
        """
        Returns labels in the native range of the chosen activation.

        Sigmoid → keeps {0, 1}   (output ∈ (0,1), target ∈ {0,1})
        Tanh    → maps  {0, 1} → {−1, +1}  (output ∈ (−1,1))

        Args:
            y_binary: integer array of 0s and 1s, shape (n,)
        Returns:
            y_native: float array suitable for error computation
        """
        if self.activation == 'tanh':
            # 0  →  −1,   1  →  +1
            return 2.0 * y_binary.astype(float) - 1.0
        else:
            return y_binary.astype(float)


    # -----------------------------------------------------------------
    # COST FUNCTION
    # -----------------------------------------------------------------

    def _calculate_cost(self, X, y_native):
        """
        Calculates Mean Squared Error across all samples.
        Formula: J(w) = (1/n) Σ (y_native − output)²

        This is the objective GD is minimising.
        Lower cost → better weights → more accurate classifier.

        Args:
            X:        feature matrix, shape (n, 4)
            y_native: targets in activation-native range
        Returns:
            cost: scalar float
        """
        z      = np.dot(X, self.weights) + self.bias
        output = self._activate(z)
        cost   = np.mean((y_native - output) ** 2)
        return cost


    # -----------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------

    def train(self, X_train, y_binary, verbose=False):
        """
        Trains using Batch Gradient Descent (delta rule).

        Per-epoch steps:
          1. Forward pass  — compute output for ALL samples at once
          2. Error         — error = y_native − output
          3. Delta         — delta = error × activation_derivative(output)
          4. Gradients     — grad_w = −(X.T @ delta) / n
                             grad_b = −mean(delta)
          5. Weight update — w ← w − lr × grad_w   (gradient descent)
                             b ← b − lr × grad_b

        This is pure batch GD: one weight update per epoch using the
        average gradient across all n training samples.

        Args:
            X_train:  feature matrix, shape (120, 4)
            y_binary: binary labels {0,1} for this OvA classifier
            verbose:  print progress every 100 epochs if True
        """

        num_samples  = X_train.shape[0]   # 120
        num_features = X_train.shape[1]   # 4

        # --- FIX BUG 1: unique seed per classifier ---
        # Using the same seed (42) for every classifier caused all
        # three OvA classifiers to start from *identical* weights,
        # meaning they explored the same loss landscape and their
        # final hyperplanes were nearly identical.  Using class_idx
        # as an offset gives each classifier a distinct starting point.
        np.random.seed(42 + self.class_idx)
        self.weights = np.random.uniform(-0.1, 0.1, num_features)
        self.bias    = 0.0

        self.cost_history     = []
        self.accuracy_history = []

        # --- FIX BUG 2: convert labels to activation-native range ---
        y_native = self._convert_labels(y_binary)

        # --- FIX BUG 3: scale epoch budget inversely with learning rate ---
        # The cost-change convergence check (original: abs_change < 1e-4,
        # patched: abs_change < 1e-6) is fundamentally lr-dependent:
        #
        #   lr=0.1  → each step moves cost by ~1e-3 to ~1e-4  (check fires late)
        #   lr=0.001→ each step moves cost by ~1e-5 to ~1e-6  (check fires at
        #             epoch 100, locking in ~44% accuracy instead of ~70%)
        #
        # No single absolute threshold can work across a 500× range of lrs.
        # The correct solution: remove the cost-change check entirely and instead
        # give each classifier an epoch budget proportional to 1/lr, so slow
        # learners get enough steps to reach their best attainable accuracy.
        #
        # Formula: effective_epochs = min(base / lr, max_cap)
        #   base    = self.epochs (the value passed in, typically 1000)
        #   max_cap = 20000  (prevents runaway runtimes for very small lr)
        #
        # Examples with self.epochs=1000:
        #   lr=0.5   → 2000 epochs
        #   lr=0.1   → 10000 epochs  (but perfect-accuracy check fires at ~459)
        #   lr=0.01  → 20000 epochs  (capped)
        #   lr=0.001 → 20000 epochs  (capped; lr=0.001 is genuinely too slow,
        #              70% is its correct converged accuracy — not a code bug)
        max_cap         = 20000
        effective_epochs = min(int(self.epochs / self.learning_rate), max_cap)

        # --- Training loop ---
        for epoch in range(effective_epochs):

            # =========================================================
            # FORWARD PASS — vectorised over all samples
            # z shape: (n,)  one weighted sum per sample
            # output shape: (n,)  one activation value per sample
            # =========================================================
            z      = np.dot(X_train, self.weights) + self.bias
            output = self._activate(z)

            # =========================================================
            # ERROR
            # error shape: (n,)
            #   positive → predicted too low   → push weights up
            #   negative → predicted too high  → push weights down
            # =========================================================
            error = y_native - output

            # =========================================================
            # DELTA RULE
            # delta = error × σ'(output)
            # Combines how wrong we are with how much the activation
            # is still in its sensitive (non-saturated) region.
            # =========================================================
            activation_deriv = self._derivative(output)
            delta = error * activation_deriv   # shape (n,)

            # =========================================================
            # BATCH GRADIENTS
            # dCost/dw_j = −(1/n) Σ_i [delta_i * x_ij]
            #            = −(X.T @ delta) / n
            # The minus sign: increasing a weight that has positive
            # delta would *reduce* cost, so the gradient of cost is
            # the negative of what we computed.
            # =========================================================
            grad_weights = -np.dot(X_train.T, delta) / num_samples
            grad_bias    = -np.mean(delta)

            # =========================================================
            # WEIGHT UPDATE — gradient descent step
            # w ← w − lr × ∇J(w)
            # Moving opposite to the gradient = moving downhill on J.
            # =========================================================
            self.weights -= self.learning_rate * grad_weights
            self.bias    -= self.learning_rate * grad_bias

            # =========================================================
            # RECORD metrics for this epoch
            # Cost uses y_native; accuracy uses original 0/1 labels.
            # =========================================================
            cost     = self._calculate_cost(X_train, y_native)
            accuracy = self._calculate_accuracy(X_train, y_binary)

            self.cost_history.append(cost)
            self.accuracy_history.append(accuracy)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1:4d} | "
                      f"Cost: {cost:.6f} | "
                      f"Accuracy: {accuracy:.4f}")

            # =========================================================
            # CONVERGENCE CHECK — perfect accuracy only
            #
            # The cost-change check has been removed entirely because
            # it is lr-dependent: with lr=0.001 the per-step cost
            # movement is ~1e-5, which is smaller than any practical
            # threshold and causes false convergence after just 100
            # epochs.  There is no single absolute threshold that works
            # across lr values spanning 0.001 to 0.5.
            #
            # Instead we rely solely on the perfect-accuracy check:
            # if the classifier has been 100% correct on the training
            # set for 20 consecutive epochs it has genuinely converged
            # for linearly-separable classes (e.g. Setosa).
            # For non-separable classes (Versicolor vs Virginica) the
            # check never fires, so training runs the full epoch budget
            # computed above, giving those classifiers every step they
            # need to find the best attainable decision boundary.
            # =========================================================
            if len(self.accuracy_history) >= 20:
                if all(acc == 1.0
                       for acc in self.accuracy_history[-20:]):
                    if verbose:
                        print(f"  Converged at epoch {epoch+1} "
                              f"(perfect accuracy for 20 epochs)")
                    break


    # -----------------------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------------------

    def predict(self, X):
        """
        Predicts binary class membership using frozen weights.

        Returns both a hard 0/1 prediction and the raw weighted sum z.
        The raw z is used by OvAClassifier for tie-breaking when
        multiple binary classifiers fire 1 simultaneously.

        Args:
            X: feature matrix, shape (n, 4)
        Returns:
            predictions: array of 0s and 1s, shape (n,)
            raw_z:       pre-activation weighted sum, shape (n,)
        """
        z           = np.dot(X, self.weights) + self.bias
        output      = self._activate(z)
        predictions = self._threshold(output)   # FIX BUG 4 (threshold correct for tanh)
        return predictions, z


    # -----------------------------------------------------------------
    # ACCURACY
    # -----------------------------------------------------------------

    def _calculate_accuracy(self, X, y_binary):
        """
        Calculates the fraction of correct binary predictions.
        Always uses 0/1 labels regardless of activation.

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