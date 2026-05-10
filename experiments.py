# =============================================================
# experiments.py
# Runs all experiments for the project:
# - Different activation functions (Task 3)
# - Different learning rates (Task 5)
# - Both algorithms compared (Task 6)
# Returns results dict for plotting and documentation.
# =============================================================

from ova import OvAClassifier
from evaluation import print_detailed_results


def run_all_experiments(X_train, y_train, X_test, y_test,
                        class_names):
    """
    Runs all required experiments and collects results.

    Experiments:
    1. Perceptron with step, sigmoid, tanh
    2. Gradient Descent with sigmoid, tanh
    3. Learning rate sweep for both algorithms

    Args:
        X_train, y_train: training data
        X_test,  y_test:  testing data
        class_names:      class name list

    Returns:
        results:          dict {experiment_name: (train_acc, test_acc)}
        ova_perceptrons:  list of trained perceptron OvA instances
        ova_gd_list:      list of trained GD OvA instances
        exp_titles_p:     titles matching ova_perceptrons
        exp_titles_gd:    titles matching ova_gd_list
    """

    results         = {}
    ova_perceptrons = []
    ova_gd_list     = []
    exp_titles_p    = []
    exp_titles_gd   = []

    # ==============================================================
    # EXPERIMENT SET 1: Perceptron — different activation functions
    # ==============================================================
    print("\n" + "=" * 55)
    print("EXPERIMENT SET 1: Perceptron Activation Functions")
    print("=" * 55)

    perceptron_activations = ['step', 'sigmoid', 'tanh']

    for activation in perceptron_activations:
        title = f"Perceptron ({activation}), lr=0.1"
        print(f"\nRunning: {title}")

        ova = OvAClassifier(
            algorithm='perceptron',
            learning_rate=0.1,
            epochs=1000,
            activation=activation
        )
        ova.train(X_train, y_train, verbose=True)

        train_acc, test_acc = print_detailed_results(
            ova, X_train, y_train, X_test, y_test,
            class_names, title
        )

        results[title]  = (train_acc, test_acc)
        ova_perceptrons.append(ova)
        exp_titles_p.append(title)

    # ==============================================================
    # EXPERIMENT SET 2: Gradient Descent — different activations
    # ==============================================================
    print("\n" + "=" * 55)
    print("EXPERIMENT SET 2: Gradient Descent Activation Functions")
    print("=" * 55)

    gd_activations = ['sigmoid', 'tanh']

    for activation in gd_activations:
        title = f"GradientDescent ({activation}), lr=0.1"
        print(f"\nRunning: {title}")

        ova = OvAClassifier(
            algorithm='gradient_descent',
            learning_rate=0.1,
            epochs=1000,
            activation=activation
        )
        ova.train(X_train, y_train, verbose=True)

        train_acc, test_acc = print_detailed_results(
            ova, X_train, y_train, X_test, y_test,
            class_names, title
        )

        results[title] = (train_acc, test_acc)
        ova_gd_list.append(ova)
        exp_titles_gd.append(title)

    # ==============================================================
    # EXPERIMENT SET 3: Learning rate sweep — Perceptron (step)
    # ==============================================================
    print("\n" + "=" * 55)
    print("EXPERIMENT SET 3: Learning Rate Sweep — Perceptron")
    print("=" * 55)

    learning_rates = [0.001, 0.01, 0.1, 0.5]

    for lr in learning_rates:
        title = f"Perceptron (step), lr={lr}"
        print(f"\nRunning: {title}")

        ova = OvAClassifier(
            algorithm='perceptron',
            learning_rate=lr,
            epochs=1000,
            activation='step'
        )
        ova.train(X_train, y_train, verbose=True)

        train_acc, test_acc = print_detailed_results(
            ova, X_train, y_train, X_test, y_test,
            class_names, title
        )

        results[title] = (train_acc, test_acc)

    # ==============================================================
    # EXPERIMENT SET 4: Learning rate sweep — Gradient Descent
    # ==============================================================
    print("\n" + "=" * 55)
    print("EXPERIMENT SET 4: Learning Rate Sweep — Gradient Descent")
    print("=" * 55)

    for lr in learning_rates:
        title = f"GradientDescent (sigmoid), lr={lr}"
        print(f"\nRunning: {title}")

        ova = OvAClassifier(
            algorithm='gradient_descent',
            learning_rate=lr,
            epochs=1000,
            activation='sigmoid'
        )
        ova.train(X_train, y_train, verbose=True)

        train_acc, test_acc = print_detailed_results(
            ova, X_train, y_train, X_test, y_test,
            class_names, title
        )

        results[title] = (train_acc, test_acc)

    return (results, ova_perceptrons, ova_gd_list,
            exp_titles_p, exp_titles_gd)