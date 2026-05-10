from preprocessing import load_and_prepare_data

X_train, X_test, y_train, y_test, class_names = load_and_prepare_data()

print("X_train shape:", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape: ", y_test.shape)
print()
print("First training flower features:", X_train[0].round(3))
print("First training flower label:   ", y_train[0], "=", class_names[y_train[0]])
print()
print("Label distribution in training:")
for i, name in enumerate(class_names):
    count = (y_train == i).sum()
    print(f"  {name}: {count} flowers")