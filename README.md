# Module 4 — Learning in AI
## AL-2002 Project 2026 | NUCES Chiniot-Faisalabad Campus
**Student:** F240544

---

## Project Overview

This project implements and compares two fundamental machine learning algorithms from scratch:

- **Perceptron Learning Rule** — binary classifier using a reactive weight update rule
- **Gradient Descent Delta Rule** — binary classifier that explicitly minimizes a cost function

Both algorithms are applied to the **Iris flower dataset** (150 samples, 3 classes, 4 features) using a **One vs All (OvA)** multi-class strategy. The project experiments with different activation functions and learning rates, documenting how each affects accuracy, convergence speed, and generalization.

---

## File Structure

```
module_4/
│
├── preprocessing.py        # Data loading, normalization, 80/20 split
├── perceptron.py           # Perceptron Learning Rule (step, sigmoid, tanh)
├── gradient_descent.py     # Gradient Descent Delta Rule (sigmoid, tanh)
├── ova.py                  # One vs All multi-class wrapper
├── evaluation.py           # Accuracy reporting and all plot generation
├── experiments.py          # Runs all 4 experiment sets
├── main.py                 # Entry point — runs everything end to end
│
├── plots_perceptron_accuracy.png   # Generated: perceptron accuracy curves
├── plots_gd_accuracy.png           # Generated: GD accuracy curves
├── plots_gd_cost.png               # Generated: GD cost curves
├── plots_comparison_bar.png        # Generated: train vs test bar chart
│
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2. Run the full experiment suite
```bash
python main.py
```

This will:
- Load and normalize the Iris dataset
- Run all 4 experiment sets (activation functions + learning rate sweeps)
- Print detailed results for every experiment
- Save 4 plot files to the working directory
- Print a final summary table

---

## Dataset

**Source:** UCI Machine Learning Repository — Iris Dataset  
**Link:** https://archive.ics.uci.edu/dataset/53/iris  
**Loaded via:** `sklearn.datasets.load_iris()` (no manual download needed)

| Property | Value |
|---|---|
| Total samples | 150 |
| Classes | Setosa (0), Versicolor (1), Virginica (2) |
| Features | Sepal length, Sepal width, Petal length, Petal width |
| Training set | 120 flowers (80%) |
| Testing set | 30 flowers (20%) |
| Normalization | Min-max scaling to [0, 1] per feature |

---

## Algorithms Implemented

### Perceptron Learning Rule
- **Update rule:** `w = w + η × (y − ŷ) × x`
- **Activation functions tested:** Step, Sigmoid, Tanh
- **Strategy:** One-flower-at-a-time weight updates (online learning)
- **Convergence:** Guaranteed only for linearly separable data
- **Multi-class:** OvA with 3 binary classifiers

### Gradient Descent Delta Rule
- **Update rule:** `w = w − η × gradient_of_MSE`
- **Activation functions tested:** Sigmoid, Tanh
- **Strategy:** Batch gradient descent — accumulates gradients over all 120 flowers, updates weights once per epoch
- **Cost function:** Mean Squared Error (MSE)
- **Convergence:** Explicitly minimizes cost function; epoch budget scales inversely with learning rate
- **Multi-class:** OvA with 3 binary classifiers

---

## Experiment Results Summary

### Experiment Set 1 — Perceptron Activation Functions (lr=0.1)

| Experiment | Train Acc | Test Acc | Gap |
|---|---|---|---|
| Perceptron (step), lr=0.1 | 94.2% | 100.0% | 0.058 |
| Perceptron (sigmoid), lr=0.1 | 94.2% | 100.0% | 0.058 |
| Perceptron (tanh), lr=0.1 | 94.2% | 100.0% | 0.058 |

**Observation:** All three activation functions produce identical results for the perceptron. This is expected — the perceptron update rule uses the thresholded binary output (0 or 1) to compute error regardless of activation function, so weight updates are identical across all three.

---

### Experiment Set 2 — Gradient Descent Activation Functions (lr=0.1)

| Experiment | Train Acc | Test Acc | Gap |
|---|---|---|---|
| GradientDescent (sigmoid), lr=0.1 | 90.0% | 96.7% | 0.067 |
| GradientDescent (tanh), lr=0.1 | 93.3% | 96.7% | 0.033 |

**Observation:** Tanh converges faster (Setosa: 48 epochs vs 459 for sigmoid) and achieves higher training accuracy. This is because tanh is zero-centered, producing stronger gradient signals and more balanced weight updates. Labels are correctly mapped to {−1, +1} for tanh to match its output range.

---

### Experiment Set 3 — Learning Rate Sweep (Perceptron, step)

| Learning Rate | Train Acc | Test Acc | Gap |
|---|---|---|---|
| lr=0.001 | 95.8% | 100.0% | 0.042 |
| lr=0.01 | 95.0% | 100.0% | 0.050 |
| lr=0.1 | 94.2% | 100.0% | 0.058 |
| lr=0.5 | 91.7% | 86.7% | 0.050 |

**Observation:** Small learning rates (0.001–0.1) all achieve 100% test accuracy. lr=0.5 overshoot — it makes corrections too aggressively, damaging weights that were already correct. This confirms the theory that large learning rates cause instability and hurt generalization.

---

### Experiment Set 4 — Learning Rate Sweep (Gradient Descent, sigmoid)

| Learning Rate | Train Acc | Test Acc | Gap | Notes |
|---|---|---|---|---|
| lr=0.001 | 65.8% | 70.0% | 0.042 | Too slow — theoretically correct |
| lr=0.01 | 82.5% | 83.3% | 0.008 | Converging, needs more epochs |
| lr=0.1 | 90.0% | 96.7% | 0.067 | Sweet spot |
| lr=0.5 | 90.0% | 96.7% | 0.067 | Fast convergence, same result |

**Observation:** lr=0.001 is genuinely too slow — even with a scaled epoch budget it cannot reach the quality of lr=0.1 within a practical budget. This is a theoretically correct observation: a learning rate 100× smaller needs 100× more epochs to cover the same distance in weight space. The sweet spot for GD on this dataset is lr=0.1.

---

## Answers to Project Questions

### Q1 — Key differences between Perceptron and Gradient Descent Delta Rule

| Property | Perceptron | Gradient Descent |
|---|---|---|
| Update trigger | After each individual flower | After all flowers (batch) |
| Error signal | Hard binary: −1, 0, or +1 | Continuous gradient value |
| Cost function | Not explicitly minimized | Explicitly minimizes MSE |
| Convergence guarantee | Only if linearly separable | Guaranteed for convex cost |
| Speed per epoch | Faster (online updates) | Slower (batch accumulation) |
| Handles overlapping classes | Struggles, may not converge | Finds best possible solution |

The perceptron reacts to individual mistakes immediately. Gradient descent looks at the total cost across all examples and moves weights systematically downhill. This makes GD more principled but also more computationally expensive per epoch.

---

### Q2 — How activation function influences performance

**For Perceptron:** Activation function choice has no effect on accuracy because the update rule uses the thresholded binary output (0 or 1), not the raw activation value. Step, sigmoid, and tanh all produce identical results.

**For Gradient Descent:** Activation function critically affects performance.
- **Sigmoid** outputs (0, 1) and matches binary {0,1} labels naturally. Gradient = output × (1 − output), which vanishes near 0 and 1, slowing learning when the model is very confident or very wrong.
- **Tanh** outputs (−1, +1). When used with correctly mapped labels {−1, +1}, it produces stronger gradients (derivative = 1 − output²) and zero-centered outputs, leading to faster convergence and slightly higher training accuracy. GD with tanh converged in 48 epochs for Setosa vs 459 for sigmoid.

---

### Q3 — Strategies for adjusting learning rate

1. **Start with a moderate value** (0.1) and observe the accuracy/cost curve over epochs.
2. **If training is unstable** (accuracy bouncing up and down): reduce the learning rate. lr=0.5 caused instability in the perceptron.
3. **If training is too slow** (barely improving after many epochs): increase the learning rate. lr=0.001 for GD barely reached 70% accuracy within its budget.
4. **Scale epoch budget inversely with learning rate**: a model with lr=0.001 needs roughly 100× more epochs than one with lr=0.1 to cover the same distance in weight space.
5. **Plot cost vs epochs** for GD experiments: a smooth monotonically decreasing curve indicates a good learning rate. A flat or erratic curve indicates the rate needs adjustment.

From our experiments, lr=0.1 was the sweet spot for both algorithms on this dataset.

---

### Q4 — Implications of different train/test split ratios

The 80/20 split (120 training, 30 testing) used in this project is the standard balance.

- **Too much training data (e.g. 95/5):** The model learns more but the evaluation is unreliable — 7 or 8 test flowers means each wrong prediction swings accuracy by 12–14%. Results are not statistically meaningful.
- **Too little training data (e.g. 50/50):** The model has fewer examples to learn from, reducing accuracy. But the larger test set gives a more reliable estimate of generalization.
- **80/20 is standard** because it gives the model enough data to learn meaningful patterns while keeping a test set large enough (30 flowers, 10 per class) for a statistically reasonable accuracy measurement.
- **Class balance within the split matters** — `random_state=42` with shuffling ensured approximately equal class representation: 40 Setosa, 41 Versicolor, 39 Virginica in training.

---

### Q5 — Challenges encountered and how they were overcome

**Challenge 1 — Identical starting weights for all 3 classifiers**  
All three OvA classifiers were initialized with `np.random.seed(42)`, giving them identical weight vectors. They explored the same loss landscape from the same starting point.  
*Fix:* Each classifier now seeds with `42 + class_idx` (42, 43, 44) for distinct initialization.

**Challenge 2 — Tanh label mismatch**  
Tanh outputs live in (−1, +1) but training labels were {0, 1}. This caused a permanent bias in the error signal — a correct prediction of −0.9 against a label of 0 still produced a nonzero error, pushing weights in the wrong direction.  
*Fix:* A `_convert_labels()` method maps {0,1} → {−1,+1} before gradient computation when tanh is selected.

**Challenge 3 — Premature convergence for small learning rates**  
The cost-change convergence check (`abs(cost[t] − cost[t−1]) < threshold`) is learning-rate-dependent. With lr=0.001, cost moves ~1e-5 per epoch, so even a generous threshold of 1e-4 caused the check to fire after only 50 epochs, locking in 44% accuracy.  
*Fix:* The cost-change check was removed entirely. The epoch budget is now scaled inversely with learning rate (`min(base_epochs / lr, 20000)`), giving slow learners proportionally more time to converge.

**Challenge 4 — Non-convergence of Versicolor/Virginica classifiers**  
These two classes overlap in feature space — no linear boundary perfectly separates them. The perceptron with step function never converges on these classifiers, bouncing chaotically across 1000 epochs.  
*Observation documented rather than fixed:* This is the theoretically correct behavior. GD with sigmoid/tanh finds a practical minimum even without perfect separation, while the perceptron cannot.

---

### Q6 — Strengths and limitations of each algorithm

**Perceptron Learning Rule**

| Strengths | Limitations |
|---|---|
| Simple to implement and understand | Cannot handle non-linearly separable data |
| Fast per-epoch (online updates) | Convergence not guaranteed for overlapping classes |
| Converges in very few epochs when data is separable (Setosa: 10 epochs) | Activation function has no effect — update rule is too blunt |
| Good test accuracy on this dataset (up to 100%) | Accuracy bounces chaotically for hard classes |

**Gradient Descent Delta Rule**

| Strengths | Limitations |
|---|---|
| Explicitly minimizes a cost function — more principled | Much slower convergence (Setosa: 459 epochs vs 10 for perceptron) |
| Handles non-linearly separable data by finding best possible solution | Sensitive to learning rate — too small means very slow learning |
| Activation function meaningfully affects learning (tanh vs sigmoid) | Batch mode means one weight update per epoch |
| Smooth, monotonically decreasing cost curves | Epoch budget must be scaled with learning rate |

**Overall conclusion:** For linearly separable data the perceptron is faster and equally accurate. For overlapping classes (Versicolor vs Virginica) gradient descent is more principled and achieves better practical accuracy. The choice of algorithm depends on the nature of the data.

---

## Implementation Notes

- All algorithms implemented **from scratch** using only NumPy (no sklearn for learning)
- sklearn used only for: `train_test_split` and `load_iris`
- Weights initialized with `np.random.uniform(-0.1, 0.1)` with per-classifier seeds
- Training data shuffled before each epoch to prevent ordering bias
- OvA conflict resolution: highest raw z-score wins when multiple classifiers fire

---

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```