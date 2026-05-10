# =============================================================
# evaluation.py
# Handles accuracy reporting and plotting.
# Generates all graphs needed for project documentation.
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 100

def print_detailed_results(ova_classifier, X_train, y_train,
                            X_test, y_test, class_names, title):
    """
    Prints a full breakdown of results for one OvA classifier.
    Shows per-class binary accuracy and overall multiclass accuracy.

    Args:
        ova_classifier: trained OvAClassifier instance
        X_train, y_train: training data
        X_test,  y_test:  testing data
        class_names:      ['setosa', 'versicolor', 'virginica']
        title:            label for this experiment
    """
    print()
    print("=" * 55)
    print(f"RESULTS: {title}")
    print("=" * 55)

    # Per-class binary classifier results
    print("\nPer-class binary classifier accuracy:")
    for i, clf in enumerate(ova_classifier.classifiers):
        epochs_run = len(clf.accuracy_history)
        final_acc  = clf.accuracy_history[-1]
        print(f"  {class_names[i]:12s} classifier: "
              f"accuracy={final_acc:.4f}, epochs={epochs_run}")

    # Overall multiclass accuracy
    train_acc = ova_classifier.score(X_train, y_train)
    test_acc  = ova_classifier.score(X_test,  y_test)

    print(f"\nOverall training accuracy: {train_acc:.4f} "
          f"({train_acc*100:.1f}%)")
    print(f"Overall testing  accuracy: {test_acc:.4f} "
          f"({test_acc*100:.1f}%)")
    print(f"Generalization gap:        "
          f"{abs(train_acc - test_acc):.4f}")

    if abs(train_acc - test_acc) > 0.1:
        print("  WARNING: Large gap suggests overfitting")
    else:
        print("  Good generalization (gap < 0.1)")

    print("=" * 55)

    return train_acc, test_acc


def plot_accuracy_curves(ova_classifiers, titles, class_names,
                         save_path=None):
    """
    Plots accuracy over epochs for each classifier in each experiment.
    One row per OvA experiment, one column per class.

    Args:
        ova_classifiers: list of trained OvAClassifier instances
        titles:          list of experiment names
        class_names:     ['setosa', 'versicolor', 'virginica']
        save_path:       if given, saves plot to this file path
    """
    num_experiments = len(ova_classifiers)
    num_classes     = 3

    fig, axes = plt.subplots(
        num_experiments, num_classes,
        figsize=(13, 3.5 * num_experiments)
    )

    # Handle case of single experiment (axes not nested)
    if num_experiments == 1:
        axes = [axes]

    for exp_idx, (ova, title) in enumerate(
            zip(ova_classifiers, titles)):

        for class_idx in range(num_classes):
            ax  = axes[exp_idx][class_idx]
            clf = ova.classifiers[class_idx]

            ax.plot(clf.accuracy_history, color='steelblue',
                    linewidth=1.5)
            ax.set_title(
                f"{title}\n{class_names[class_idx]} classifier",
                fontsize=10
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Binary Accuracy")
            ax.set_ylim(0, 1.05)
            ax.axhline(y=1.0, color='green', linestyle='--',
                       alpha=0.5, label='Perfect')
            ax.grid(True, alpha=0.3)

    plt.suptitle("Accuracy Over Epochs — Per Classifier",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    plt.show()


def plot_cost_curves(ova_gd_classifiers, titles, class_names,
                     save_path=None):
    """
    Plots cost over epochs for gradient descent classifiers only.
    Cost curves show how MSE decreases during training.

    Args:
        ova_gd_classifiers: list of trained GD OvAClassifier instances
        titles:             experiment names
        class_names:        class name list
        save_path:          optional save path
    """
    num_experiments = len(ova_gd_classifiers)

    fig, axes = plt.subplots(
        num_experiments, 3,
        figsize=(13, 3.5 * num_experiments)
    )

    if num_experiments == 1:
        axes = [axes]

    for exp_idx, (ova, title) in enumerate(
            zip(ova_gd_classifiers, titles)):

        for class_idx in range(3):
            ax  = axes[exp_idx][class_idx]
            clf = ova.classifiers[class_idx]

            ax.plot(clf.cost_history, color='crimson',
                    linewidth=1.5)
            ax.set_title(
                f"{title}\n{class_names[class_idx]} cost",
                fontsize=10
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE Cost")
            ax.grid(True, alpha=0.3)

    plt.suptitle("Cost Over Epochs — Gradient Descent",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    plt.show()


def plot_comparison_bar(results_dict, save_path=None):
    """
    Bar chart comparing train vs test accuracy across experiments.
    Easy to include in Word document for documentation.

    Args:
        results_dict: dict of {experiment_name: (train_acc, test_acc)}
        save_path:    optional save path
    """
    labels      = list(results_dict.keys())
    train_accs  = [v[0] for v in results_dict.values()]
    test_accs   = [v[1] for v in results_dict.values()]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_train = ax.bar(x - width/2, train_accs, width,
                        label='Training Accuracy', color='steelblue',
                        alpha=0.85)
    bars_test  = ax.bar(x + width/2, test_accs,  width,
                        label='Testing Accuracy',  color='darkorange',
                        alpha=0.85)

    # Add value labels on top of bars
    for bar in bars_train:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha='center', va='bottom', fontsize=8)
    for bar in bars_test:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha='center', va='bottom', fontsize=8)

    ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Testing Accuracy — All Experiments",
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Plot saved: {save_path}")
    plt.show()