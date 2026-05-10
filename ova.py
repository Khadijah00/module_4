# =============================================================
# ova.py
# One vs All (OvA) wrapper for multi-class classification.
# Trains one binary classifier per class.
# At prediction time, combines all 3 classifiers to decide
# the final flower species.
# Works with both Perceptron and GradientDescent classes.
#
# FIX: _make_classifier() now passes class_idx to each
#      GradientDescent instance so that each of the 3 binary
#      classifiers is seeded differently and starts from
#      distinct initial weights.  (Perceptron already shuffles
#      its training data each epoch, so it is less affected,
#      but we pass class_idx there too for full correctness.)
# =============================================================

import numpy as np
from perceptron import Perceptron
from gradient_descent import GradientDescent


class OvAClassifier:
    """
    One-vs-All multi-class classifier for the 3 Iris species.

    Trains 3 independent binary classifiers:
      Classifier 0: Is it Setosa?      (y=1 if Setosa,     else 0)
      Classifier 1: Is it Versicolor?  (y=1 if Versicolor,  else 0)
      Classifier 2: Is it Virginica?   (y=1 if Virginica,   else 0)

    Prediction combines the three outputs:
      - Exactly one fires 1  → that class is the winner
      - Multiple fire 1      → highest raw z-score wins
      - None fires 1         → least-negative z-score wins

    Attributes:
        algorithm:          'perceptron' or 'gradient_descent'
        learning_rate:      passed to each binary classifier
        epochs:             passed to each binary classifier
        activation:         passed to each binary classifier
        classifiers:        list of 3 trained binary classifiers
        accuracy_histories: per-class accuracy-over-epochs lists
    """

    def __init__(self, algorithm='perceptron', learning_rate=0.1,
                 epochs=1000, activation='step'):
        """
        Initialises OvA with chosen algorithm and hyperparameters.

        Args:
            algorithm:     'perceptron' or 'gradient_descent'
            learning_rate: passed through to each binary classifier
            epochs:        passed through to each binary classifier
            activation:    passed through to each binary classifier
        """
        self.algorithm     = algorithm
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.activation    = activation
        self.classifiers   = []
        self.accuracy_histories = []


    def _make_classifier(self, class_idx):
        """
        Creates a fresh binary classifier for the given class index.

        The class_idx is forwarded to GradientDescent (and Perceptron)
        so each classifier starts with a unique random seed.  Without
        this, all three classifiers would begin from identical weights,
        which causes them to converge to nearly the same hyperplane.

        Args:
            class_idx: 0, 1, or 2  (Setosa / Versicolor / Virginica)
        Returns:
            A fresh, untrained binary classifier instance.
        """
        if self.algorithm == 'perceptron':
            return Perceptron(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation,
                class_idx=class_idx
            )
        elif self.algorithm == 'gradient_descent':
            return GradientDescent(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation,
                class_idx=class_idx   # FIX: unique seed per classifier
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


    def train(self, X_train, y_train, verbose=False):
        """
        Trains 3 independent binary classifiers using the OvA strategy.

        For class k:  y_binary[i] = 1 if y_train[i] == k, else 0.
        Each classifier sees the full training set but with different
        binary labels, so they learn three separate decision boundaries.

        Args:
            X_train: feature matrix, shape (120, 4)
            y_train: multi-class labels {0, 1, 2}, shape (120,)
            verbose: print per-classifier training progress if True
        """
        self.classifiers        = []
        self.accuracy_histories = []

        class_names = ['Setosa', 'Versicolor', 'Virginica']

        for class_idx in range(3):

            if verbose:
                print(f"  Training classifier {class_idx+1}/3: "
                      f"{class_names[class_idx]}?")

            # Build binary target: 1 for this class, 0 for all others
            y_binary = (y_train == class_idx).astype(int)

            # Create and train a fresh classifier with unique seed
            clf = self._make_classifier(class_idx)
            clf.train(X_train, y_binary, verbose=False)

            self.classifiers.append(clf)
            self.accuracy_histories.append(clf.accuracy_history)

            if verbose:
                final_acc  = clf.accuracy_history[-1]
                epochs_run = len(clf.accuracy_history)
                print(f"    Done. Epochs: {epochs_run}, "
                      f"Binary accuracy: {final_acc:.4f}")


    def predict(self, X):
        """
        Predicts flower species by combining the 3 binary classifiers.

        Decision rules (in priority order):
          1. Exactly one classifier fires 1  → that class wins
          2. Multiple classifiers fire 1     → highest raw z-score wins
          3. No classifier fires 1           → least-negative z-score wins

        Rules 2 and 3 use the same argmax logic, so they share a branch.

        Args:
            X: feature matrix, shape (n, 4)
        Returns:
            final_predictions: class indices {0,1,2}, shape (n,)
        """
        num_samples = X.shape[0]

        # Collect binary predictions and raw z-scores from all classifiers
        # Shape of each list element: (num_samples,)
        all_predictions = []
        all_z_scores    = []

        for clf in self.classifiers:
            preds, z = clf.predict(X)
            all_predictions.append(preds)
            all_z_scores.append(z)

        # Stack into 2-D arrays: shape (3, num_samples)
        predictions = np.array(all_predictions)
        z_scores    = np.array(all_z_scores)

        final_predictions = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):

            flower_preds  = predictions[:, i]   # [p0, p1, p2]
            flower_z      = z_scores[:, i]       # [z0, z1, z2]

            fires = np.sum(flower_preds)

            if fires == 1:
                # Unambiguous: exactly one classifier detected its class
                final_predictions[i] = np.argmax(flower_preds)
            else:
                # Conflict (multiple fire) or silence (none fire):
                # use raw z-scores to resolve — highest z wins.
                # For conflicts this picks the most-confident firing.
                # For silence this picks the least-wrong classifier.
                final_predictions[i] = np.argmax(flower_z)

        return final_predictions


    def score(self, X, y_true):
        """
        Computes overall multi-class accuracy on the given dataset.

        Args:
            X:      feature matrix
            y_true: true class labels {0, 1, 2}
        Returns:
            accuracy: float in [0, 1]
        """
        predictions = self.predict(X)
        correct     = np.sum(predictions == y_true)
        accuracy    = correct / len(y_true)
        return accuracy