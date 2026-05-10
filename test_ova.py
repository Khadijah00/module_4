from preprocessing import load_and_prepare_data
from ova import OvAClassifier

X_train, X_test, y_train, y_test, class_names = load_and_prepare_data()

print("=" * 45)
print("TEST 1: Perceptron OvA (step function)")
print("=" * 45)
ova_p = OvAClassifier(algorithm='perceptron', learning_rate=0.1,
                      epochs=1000, activation='step')
ova_p.train(X_train, y_train, verbose=True)
train_acc = ova_p.score(X_train, y_train)
test_acc  = ova_p.score(X_test,  y_test)
print(f"\nOverall training accuracy: {train_acc:.4f}")
print(f"Overall testing accuracy:  {test_acc:.4f}")

print()
print("=" * 45)
print("TEST 2: Gradient Descent OvA (sigmoid)")
print("=" * 45)
ova_gd = OvAClassifier(algorithm='gradient_descent', learning_rate=0.1,
                       epochs=1000, activation='sigmoid')
ova_gd.train(X_train, y_train, verbose=True)
train_acc = ova_gd.score(X_train, y_train)
test_acc  = ova_gd.score(X_test,  y_test)
print(f"\nOverall training accuracy: {train_acc:.4f}")
print(f"Overall testing accuracy:  {test_acc:.4f}")