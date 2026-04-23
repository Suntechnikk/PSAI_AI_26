import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 11
Ee = 0.0001
max_epochs = 200
lr_fixed = 0.1

def generate_truth_table(n):
    X = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2 ** n)])
    y = np.any(X == 1, axis=1).astype(int).reshape(-1, 1)
    return X, y

def split_data(X, y, train_ratio=0.8):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def bce_loss(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def update_learning_rate(current_lr, prev_error, curr_error):
    if curr_error < prev_error:
        return current_lr * 1.05
    return current_lr * 0.7

def train_model(X, y, loss_type="mse", adaptive=False, lr=0.1, max_epochs=200, Ee=0.0001):
    w = np.random.randn(n, 1) * 0.1
    b = np.random.randn() * 0.1
    errors = []
    lr_current = lr

    for epoch in range(max_epochs):
        total_error = 0.0
        order = np.random.permutation(len(X))

        for i in order:
            xi = X[i].reshape(-1, 1)
            yi = y[i].reshape(1, 1)

            z = w.T @ xi + b
            y_pred = sigmoid(z)

            if loss_type == "mse":
                sample_error = mse_loss(yi, y_pred)
                delta = (y_pred - yi) * y_pred * (1 - y_pred)
            else:
                sample_error = bce_loss(yi, y_pred)
                delta = (y_pred - yi)

            w -= lr_current * xi * delta
            b -= lr_current * delta.item()
            total_error += sample_error

        total_error /= len(X)
        errors.append(total_error)

        if adaptive and epoch > 0:
            lr_current = update_learning_rate(lr_current, errors[-2], errors[-1])

        if total_error <= Ee:
            break

    return w, b, errors, epoch + 1

def predict(X, w, b):
    probs = sigmoid(X @ w + b)
    classes = (probs >= 0.5).astype(int)
    return probs, classes

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def evaluate_model(X, y, w, b):
    _, classes = predict(X, w, b)
    return accuracy_score(y, classes)

def true_or_function(x):
    return int(np.any(x == 1))

X, y = generate_truth_table(n)
X_train, X_test, y_train, y_test = split_data(X, y)

configs = [
    ("MSE fixed", "mse", False),
    ("MSE adaptive", "mse", True),
    ("BCE fixed", "bce", False),
    ("BCE adaptive", "bce", True),
]

results = {}

for name, loss_type, adaptive in configs:
    w, b, errors, epochs = train_model(
        X_train,
        y_train,
        loss_type=loss_type,
        adaptive=adaptive,
        lr=lr_fixed,
        max_epochs=max_epochs,
        Ee=Ee
    )

    train_acc = evaluate_model(X_train, y_train, w, b)
    test_acc = evaluate_model(X_test, y_test, w, b)
    full_acc = evaluate_model(X, y, w, b)

    results[name] = {
        "w": w,
        "b": b,
        "errors": errors,
        "epochs": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "full_acc": full_acc
    }

print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
print("-" * 72)
print(f"{'Метод':<18} {'Эпохи':<10} {'Train acc':<12} {'Test acc':<12} {'Full acc':<12}")
print("-" * 72)

for name, res in results.items():
    print(f"{name:<18} {res['epochs']:<10} {res['train_acc']:<12.4f} {res['test_acc']:<12.4f} {res['full_acc']:<12.4f}")

print("-" * 72)

styles = {
    "MSE fixed": {"linestyle": "-", "marker": "o"},
    "MSE adaptive": {"linestyle": "--", "marker": "s"},
    "BCE fixed": {"linestyle": "-.", "marker": "^"},
    "BCE adaptive": {"linestyle": ":", "marker": "d"}
}

plt.figure(figsize=(10, 6))

for name, res in results.items():
    epochs_range = range(1, len(res["errors"]) + 1)
    plt.plot(
        epochs_range,
        res["errors"],
        label=name,
        linestyle=styles[name]["linestyle"],
        marker=styles[name]["marker"],
        markersize=4
    )

plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Сходимость (MSE vs BCE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

best_name = None
best_res = None

for name, res in results.items():
    if best_res is None:
        best_name = name
        best_res = res
    else:
        if res["test_acc"] > best_res["test_acc"]:
            best_name = name
            best_res = res
        elif res["test_acc"] == best_res["test_acc"] and res["epochs"] < best_res["epochs"]:
            best_name = name
            best_res = res

print(f"\nЛучшая конфигурация: {best_name}")

def user_mode(w, b):
    print(f"\nВведите {n} значений (0 или 1) через пробел:")
    raw = input().strip().split()

    if len(raw) != n:
        print(f"Ошибка: нужно ввести ровно {n} значений.")
        return

    try:
        x = np.array(list(map(int, raw))).reshape(1, -1)
    except ValueError:
        print("Ошибка: ввод должен содержать только числа 0 или 1.")
        return

    if not np.all((x == 0) | (x == 1)):
        print("Ошибка: допустимы только 0 и 1.")
        return

    prob = sigmoid(x @ w + b)[0, 0]
    pred_class = int(prob >= 0.5)
    true_class = true_or_function(x)

    print(f"Вероятность: {prob:.4f}")
    print(f"Класс: {pred_class}")
    print("Совпадает с таблицей истинности" if pred_class == true_class else "Расхождение")

user_mode(best_res["w"], best_res["b"])