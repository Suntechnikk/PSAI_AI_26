import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# =========================
# 1. Генерация данных (AND, n=9)
# =========================
n = 9

def generate_data(n):
    X = np.array([list(map(int, format(i, f'0{n}b'))) for i in range(2**n)])
    y = np.all(X == 1, axis=1).astype(int)  # AND
    return X, y.reshape(-1, 1)

X, y = generate_data(n)

# =========================
# 2. train/test split
# =========================
indices = np.arange(len(X))
np.random.shuffle(indices)

split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# =========================
# 3. Модель
# =========================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# =========================
# 4. Функции потерь
# =========================
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def bce(y_true, y_pred):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# =========================
# 5. Обучение
# =========================
def train(X, y, loss_type="mse", lr=0.1, adaptive=False, epochs=1000, Ee=0.01):
    w = np.random.randn(n, 1)
    b = 0

    errors = []
    lr_current = lr

    for epoch in range(epochs):
        total_error = 0

        for i in range(len(X)):
            xi = X[i].reshape(-1, 1)
            yi = y[i]

            z = np.dot(w.T, xi) + b
            y_pred = sigmoid(z)

            if loss_type == "mse":
                error = mse(yi, y_pred)
                grad = (y_pred - yi) * y_pred * (1 - y_pred)
            else:  # BCE
                error = bce(yi, y_pred)
                grad = (y_pred - yi)

            # обновление весов
            w -= lr_current * grad * xi
            b -= lr_current * grad

            total_error += error

        total_error /= len(X)
        errors.append(total_error)

        # адаптивный шаг
        if adaptive and epoch > 0:
            if errors[-1] > errors[-2]:
                lr_current *= 0.7
            else:
                lr_current *= 1.05

        if total_error <= Ee:
            break

    return w, b, errors, epoch+1

# =========================
# 6. Оценка
# =========================
def evaluate(X, y, w, b):
    preds = sigmoid(X @ w + b)
    preds_class = (preds >= 0.5).astype(int)
    acc = np.mean(preds_class == y)
    return acc

# =========================
# 7. Запуск 4 конфигураций
# =========================
configs = [
    ("MSE fixed", "mse", False),
    ("MSE adaptive", "mse", True),
    ("BCE fixed", "bce", False),
    ("BCE adaptive", "bce", True),
]

results = {}

for name, loss, adaptive in configs:
    w, b, errors, epochs = train(X_train, y_train, loss_type=loss, adaptive=adaptive)

    train_acc = evaluate(X_train, y_train, w, b)
    test_acc = evaluate(X_test, y_test, w, b)
    full_acc = evaluate(X, y, w, b)

    results[name] = {
        "errors": errors,
        "epochs": epochs,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "full_acc": full_acc,
        "w": w,
        "b": b
    }

# =========================
# 8. График
# =========================
plt.figure(figsize=(10,6))

for name in results:
    plt.plot(results[name]["errors"], label=name)

plt.xlabel("Эпоха")
plt.ylabel("Ошибка")
plt.title("Сходимость (MSE vs BCE)")
plt.legend()
plt.grid()
plt.show()

# =========================
# 9. Результаты
# =========================
for name in results:
    r = results[name]
    print(f"\n{name}")
    print(f"Эпохи: {r['epochs']}")
    print(f"Train accuracy: {r['train_acc']:.4f}")
    print(f"Test accuracy: {r['test_acc']:.4f}")
    print(f"Full accuracy: {r['full_acc']:.4f}")

# =========================
# 10. Режим пользователя
# =========================
def predict_user(w, b):
    print("\nВведи 9 значений (0 или 1):")
    x = list(map(int, input().split()))

    x = np.array(x).reshape(1, -1)
    y_pred = sigmoid(x @ w + b)[0][0]
    y_class = int(y_pred >= 0.5)

    true = int(np.all(x == 1))

    print(f"Вероятность: {y_pred:.4f}")
    print(f"Класс: {y_class}")
    print("Совпадает" if y_class == true else "Расхождение")

# пример использования (BCE adaptive лучше всего)
best = results["BCE adaptive"]
predict_user(best["w"], best["b"])