# =============================================================
# ova.py
# One vs All (OvA) wrapper for multi-class classification.
# Trains one binary classifier per class.
# At prediction time, combines all 3 classifiers to decide
# the final flower species.
# Works with both Perceptron and GradientDescent classes.
# =============================================================

import numpy as np
from perceptron import Perceptron
from gradient_descent import GradientDescent


class OvAClassifier:
    """
    One vs All multi-class classifier.
    Trains 3 binary classifiers for 3 iris species.

    Classifier 0: Is it Setosa?     (y=1 if Setosa,     else 0)
    Classifier 1: Is it Versicolor? (y=1 if Versicolor,  else 0)
    Classifier 2: Is it Virginica?  (y=1 if Virginica,   else 0)

    Prediction: whichever classifier fires 1 wins.
    Conflict (multiple fire): highest raw z wins.
    None fire: least negative z wins.

    Attributes:
        algorithm:    'perceptron' or 'gradient_descent'
        classifiers:  list of 3 trained binary classifiers
        accuracy_histories: accuracy per epoch per classifier
    """

    def __init__(self, algorithm='perceptron', learning_rate=0.1,
                 epochs=1000, activation='step'):
        """
        Initializes OvA with chosen algorithm and hyperparameters.

        Args:
            algorithm:     'perceptron' or 'gradient_descent'
            learning_rate: passed to each binary classifier
            epochs:        passed to each binary classifier
            activation:    passed to each binary classifier
        """
        self.algorithm     = algorithm
        self.learning_rate = learning_rate
        self.epochs        = epochs
        self.activation    = activation
        self.classifiers   = []
        self.accuracy_histories = []


    def _make_classifier(self):
        """
        Creates a fresh binary classifier based on algorithm setting.
        """
        if self.algorithm == 'perceptron':
            return Perceptron(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation
            )
        elif self.algorithm == 'gradient_descent':
            return GradientDescent(
                learning_rate=self.learning_rate,
                epochs=self.epochs,
                activation=self.activation
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")


    def train(self, X_train, y_train, verbose=False):
        """
        Trains 3 binary classifiers using OvA strategy.
        Each classifier trains completely independently.

        Args:
            X_train: training features shape (120, 4)
            y_train: multi-class labels (0, 1, or 2)
            verbose: print training progress
        """
        self.classifiers        = []
        self.accuracy_histories = []

        class_names = ['Setosa', 'Versicolor', 'Virginica']

        for class_idx in range(3):

            if verbose:
                print(f"  Training classifier {class_idx+1}/3: "
                      f"{class_names[class_idx]}?")

            # Create binary labels for this classifier
            # 1 where this is the target class, 0 everywhere else
            y_binary = (y_train == class_idx).astype(int)

            # Create and train a fresh classifier
            clf = self._make_classifier()
            clf.train(X_train, y_binary, verbose=False)

            self.classifiers.append(clf)
            self.accuracy_histories.append(clf.accuracy_history)

            if verbose:
                final_acc = clf.accuracy_history[-1]
                epochs_run = len(clf.accuracy_history)
                print(f"    Done. Epochs: {epochs_run}, "
                      f"Binary accuracy: {final_acc:.4f}")


    def predict(self, X):
        """
        Predicts flower species for given inputs.
        Combines outputs from all 3 binary classifiers.

        Decision rules:
        - Exactly one fires 1  → that class wins
        - Multiple fire 1      → highest raw z wins
        - None fire 1          → least negative z wins

        Args:
            X: flower features shape (n, 4)
        Returns:
            final_predictions: class indices (0, 1, or 2)
        """
        num_flowers = X.shape[0]

        # Collect outputs and raw z from all 3 classifiers
        # Shape of each: (num_flowers,)
        all_outputs = []
        all_z_scores = []

        for clf in self.classifiers:
            predictions, z = clf.predict(X)
            all_outputs.append(predictions)
            all_z_scores.append(z)

        # Stack into arrays for easy comparison
        # Shape: (3, num_flowers)
        outputs  = np.array(all_outputs)
        z_scores = np.array(all_z_scores)

        final_predictions = np.zeros(num_flowers, dtype=int)

        for i in range(num_flowers):

            flower_outputs  = outputs[:, i]   # [out0, out1, out2]
            flower_z_scores = z_scores[:, i]  # [z0,   z1,   z2]

            # Count how many classifiers fired 1
            fires = np.sum(flower_outputs)

            if fires == 1:
                # Exactly one fired — clear winner
                final_predictions[i] = np.argmax(flower_outputs)

            else:
                # Conflict or none fired — use highest z score
                # Works for both cases:
                # conflict: highest z among those that fired
                # none fire: least negative z (closest to firing)
                final_predictions[i] = np.argmax(flower_z_scores)

        return final_predictions


    def score(self, X, y_true):
        """
        Calculates overall multi-class accuracy.

        Args:
            X:      flower features
            y_true: true class labels (0, 1, or 2)
        Returns:
            accuracy: float 0 to 1
        """
        predictions = self.predict(X)
        correct     = np.sum(predictions == y_true)
        accuracy    = correct / len(y_true)
        return accuracy