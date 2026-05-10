# =============================================================
# main.py
# Entry point for Module 4 project.
# Runs all experiments, prints results, generates all plots.
# =============================================================

from preprocessing import load_and_prepare_data
from experiments  import run_all_experiments
from evaluation   import (plot_accuracy_curves,
                           plot_cost_curves,
                           plot_comparison_bar)


def main():

    # ----------------------------------------------------------
    # Step 1: Load and prepare data
    # ----------------------------------------------------------
    X_train, X_test, y_train, y_test, class_names = \
        load_and_prepare_data()

    # ----------------------------------------------------------
    # Step 2: Run all experiments
    # ----------------------------------------------------------
    (results,
     ova_perceptrons,
     ova_gd_list,
     exp_titles_p,
     exp_titles_gd) = run_all_experiments(
        X_train, y_train, X_test, y_test, class_names
    )

    # ----------------------------------------------------------
    # Step 3: Generate plots
    # ----------------------------------------------------------
    print("\nGenerating plots...")

    # Accuracy curves for perceptron experiments
    plot_accuracy_curves(
        ova_perceptrons,
        exp_titles_p,
        class_names,
        save_path="plots_perceptron_accuracy.png"
    )

    # Accuracy curves for gradient descent experiments
    plot_accuracy_curves(
        ova_gd_list,
        exp_titles_gd,
        class_names,
        save_path="plots_gd_accuracy.png"
    )

    # Cost curves for gradient descent
    plot_cost_curves(
        ova_gd_list,
        exp_titles_gd,
        class_names,
        save_path="plots_gd_cost.png"
    )

    # Bar chart comparing all experiments
    plot_comparison_bar(
        results,
        save_path="plots_comparison_bar.png"
    )

    # ----------------------------------------------------------
    # Step 4: Final summary table
    # ----------------------------------------------------------
    print("\n" + "=" * 65)
    print("FINAL SUMMARY — ALL EXPERIMENTS")
    print("=" * 65)
    print(f"{'Experiment':<40} {'Train':>7} {'Test':>7} {'Gap':>7}")
    print("-" * 65)
    for name, (train_acc, test_acc) in results.items():
        gap = abs(train_acc - test_acc)
        print(f"{name:<40} {train_acc:>7.4f} {test_acc:>7.4f} "
              f"{gap:>7.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()