import itertools
import numpy as np
import matplotlib.pyplot as plt


# =========================
# НАСТРОЙКИ ЛАБЫ
# =========================
N_INPUTS = 9
TRAIN_RATIO = 0.7
MAX_EPOCHS = 5000
RANDOM_SEED = 42

# Фиксированные шаги обучения для эксперимента
FIXED_LEARNING_RATES = [0.1, 0.3, 0.5]

# Начальный шаг для адаптивного обучения
ADAPTIVE_INITIAL_LR = 0.5


# =========================
# ЛОГИЧЕСКАЯ ФУНКЦИЯ AND
# =========================
def logical_and(x: np.ndarray) -> int:
    """
    Логическая функция AND для вектора из 0 и 1.
    Возвращает 1, только если все элементы равны 1, иначе 0.
    """
    return int(np.all(x == 1))


# =========================
# ГЕНЕРАЦИЯ ТАБЛИЦЫ ИСТИННОСТИ
# =========================
def generate_truth_table(n: int):
    """
    Генерирует полную таблицу истинности для n входов.
    X: все возможные комбинации 0/1 длины n
    y: значения функции AND
    """
    combinations = list(itertools.product([0, 1], repeat=n))
    X = np.array(combinations, dtype=float)
    y = np.array([logical_and(row) for row in X], dtype=float).reshape(-1, 1)
    return X, y


# =========================
# РАЗДЕЛЕНИЕ НА TRAIN / TEST
# =========================
def train_test_split_manual(X, y, train_ratio=0.7, seed=42):
    """
    Перемешивает данные и делит их на обучающую и тестовую выборки.
    """
    np.random.seed(seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    split_index = int(len(X) * train_ratio)
    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, X_test, y_train, y_test


# =========================
# СИГМОИДА И ОШИБКА
# =========================
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# =========================
# ПЕРСЕПТРОН
# =========================
class SingleLayerPerceptron:
    def __init__(self, n_inputs: int):
        self.n_inputs = n_inputs
        self.weights = np.random.uniform(-0.5, 0.5, (n_inputs, 1))
        self.bias = np.random.uniform(-0.5, 0.5, (1,))

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return sigmoid(z)

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def train_fixed_lr(self, X_train, y_train, X_test, y_test, learning_rate=0.1, max_epochs=1000):
        train_errors = []
        test_errors = []

        for epoch in range(max_epochs):
            y_pred_train = self.forward(X_train)
            error = y_train - y_pred_train

            grad_output = error * y_pred_train * (1 - y_pred_train)
            grad_w = np.dot(X_train.T, grad_output) / len(X_train)
            grad_b = np.mean(grad_output, axis=0)

            self.weights += learning_rate * grad_w
            self.bias += learning_rate * grad_b

            train_pred = self.forward(X_train)
            test_pred = self.forward(X_test)

            train_loss = mse_loss(y_train, train_pred)
            test_loss = mse_loss(y_test, test_pred)

            train_errors.append(train_loss)
            test_errors.append(test_loss)

            if train_loss < 1e-5:
                break

        return train_errors, test_errors, epoch + 1

    def train_adaptive_lr(self, X_train, y_train, X_test, y_test, initial_lr=0.5, max_epochs=1000):
        train_errors = []
        test_errors = []
        learning_rate = initial_lr
        prev_train_loss = float("inf")

        for epoch in range(max_epochs):
            old_weights = self.weights.copy()
            old_bias = self.bias.copy()

            y_pred_train = self.forward(X_train)
            error = y_train - y_pred_train

            grad_output = error * y_pred_train * (1 - y_pred_train)
            grad_w = np.dot(X_train.T, grad_output) / len(X_train)
            grad_b = np.mean(grad_output, axis=0)

            self.weights += learning_rate * grad_w
            self.bias += learning_rate * grad_b

            new_train_pred = self.forward(X_train)
            train_loss = mse_loss(y_train, new_train_pred)

            if train_loss < prev_train_loss:
                learning_rate *= 1.05
            else:
                self.weights = old_weights
                self.bias = old_bias
                learning_rate *= 0.7

                self.weights += learning_rate * grad_w
                self.bias += learning_rate * grad_b
                new_train_pred = self.forward(X_train)
                train_loss = mse_loss(y_train, new_train_pred)

            test_pred = self.forward(X_test)
            test_loss = mse_loss(y_test, test_pred)

            train_errors.append(train_loss)
            test_errors.append(test_loss)
            prev_train_loss = train_loss

            if train_loss < 1e-5:
                break

        return train_errors, test_errors, epoch + 1


# =========================
# ОЦЕНКА КАЧЕСТВА
# =========================
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def print_dataset_examples(X, y, count=10):
    print("\nПервые примеры таблицы истинности:")
    for i in range(min(count, len(X))):
        print(f"{X[i].astype(int)} -> {int(y[i][0])}")


def evaluate_model(model, X_train, y_train, X_test, y_test, title=""):
    train_probs = model.predict_proba(X_train)
    train_preds = model.predict(X_train)
    test_probs = model.predict_proba(X_test)
    test_preds = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print(f"\n===== {title} =====")
    print(f"Точность на обучающей выборке: {train_acc:.4f}")
    print(f"Точность на тестовой выборке:   {test_acc:.4f}")
    print("\nИтоговые веса:")
    for i, w in enumerate(model.weights.flatten(), start=1):
        print(f"w{i} = {w:.6f}")
    print(f"bias = {model.bias[0]:.6f}")

    return train_acc, test_acc, train_probs, test_probs


# =========================
# ГРАФИКИ
# =========================
def plot_errors(train_errors, test_errors, title):
    plt.figure(figsize=(10, 6))
    plt.plot(train_errors, label="Ошибка на train")
    plt.plot(test_errors, label="Ошибка на test")
    plt.xlabel("Эпоха")
    plt.ylabel("MSE ошибка")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================
# РЕЖИМ ФУНКЦИОНИРОВАНИЯ
# =========================
def interactive_mode(model, n_inputs):
    print("\n===== РЕЖИМ ФУНКЦИОНИРОВАНИЯ =====")
    print(f"Введите {n_inputs} чисел 0 или 1 через пробел.")
    print("Чтобы выйти, введите: exit")

    while True:
        user_input = input("\nВаш вектор: ").strip()

        if user_input.lower() == "exit":
            print("Выход из режима функционирования.")
            break

        parts = user_input.split()

        if len(parts) != n_inputs:
            print(f"Ошибка: нужно ввести ровно {n_inputs} чисел.")
            continue

        try:
            vector = np.array([int(x) for x in parts], dtype=float)
        except ValueError:
            print("Ошибка: вводите только числа 0 или 1.")
            continue

        if not np.all(np.isin(vector, [0, 1])):
            print("Ошибка: допустимы только 0 и 1.")
            continue

        vector = vector.reshape(1, -1)
        probability = model.predict_proba(vector)[0][0]
        predicted_class = int(probability >= 0.5)

        print(f"Вероятность класса 1: {probability:.6f}")
        print(f"Округленный класс: {predicted_class}")


# =========================
# ОСНОВНАЯ ПРОГРАММА
# =========================
def main():
    print("Лабораторная работа №4")
    print("Классификация линейно разделимых образов однослойным персептроном")
    print(f"Вариант 18: n = {N_INPUTS}, логическая функция AND")

    X, y = generate_truth_table(N_INPUTS)
    print(f"\nВсего наборов в таблице истинности: {len(X)}")
    print_dataset_examples(X, y, count=10)

    X_train, X_test, y_train, y_test = train_test_split_manual(
        X, y, train_ratio=TRAIN_RATIO, seed=RANDOM_SEED
    )

    print(f"\nРазмер обучающей выборки: {len(X_train)}")
    print(f"Размер тестовой выборки:   {len(X_test)}")

    fixed_results = []

    for lr in FIXED_LEARNING_RATES:
        np.random.seed(RANDOM_SEED)
        model = SingleLayerPerceptron(N_INPUTS)

        train_errors, test_errors, epochs = model.train_fixed_lr(
            X_train, y_train, X_test, y_test,
            learning_rate=lr,
            max_epochs=MAX_EPOCHS
        )

        train_acc, test_acc, _, _ = evaluate_model(
            model, X_train, y_train, X_test, y_test,
            title=f"Фиксированный шаг обучения alpha = {lr}"
        )

        print(f"Количество эпох: {epochs}")

        plot_errors(
            train_errors,
            test_errors,
            title=f"Фиксированный шаг alpha = {lr}"
        )

        fixed_results.append({
            "lr": lr,
            "epochs": epochs,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "model": model,
            "train_errors": train_errors,
            "test_errors": test_errors
        })

    np.random.seed(RANDOM_SEED)
    adaptive_model = SingleLayerPerceptron(N_INPUTS)

    adaptive_train_errors, adaptive_test_errors, adaptive_epochs = adaptive_model.train_adaptive_lr(
        X_train, y_train, X_test, y_test,
        initial_lr=ADAPTIVE_INITIAL_LR,
        max_epochs=MAX_EPOCHS
    )

    adaptive_train_acc, adaptive_test_acc, _, _ = evaluate_model(
        adaptive_model, X_train, y_train, X_test, y_test,
        title=f"Адаптивный шаг обучения, начальный alpha = {ADAPTIVE_INITIAL_LR}"
    )

    print(f"Количество эпох: {adaptive_epochs}")

    plot_errors(
        adaptive_train_errors,
        adaptive_test_errors,
        title=f"Адаптивный шаг, начальный alpha = {ADAPTIVE_INITIAL_LR}"
    )

    print("\n===== ИТОГОВОЕ СРАВНЕНИЕ =====")
    for result in fixed_results:
        print(
            f"Фиксированный alpha = {result['lr']}: "
            f"эпох = {result['epochs']}, "
            f"train_acc = {result['train_acc']:.4f}, "
            f"test_acc = {result['test_acc']:.4f}"
        )

    print(
        f"Адаптивный шаг: эпох = {adaptive_epochs}, "
        f"train_acc = {adaptive_train_acc:.4f}, "
        f"test_acc = {adaptive_test_acc:.4f}"
    )

    best_fixed = max(fixed_results, key=lambda r: r["test_acc"])
    best_model = adaptive_model if adaptive_test_acc >= best_fixed["test_acc"] else best_fixed["model"]

    if best_model is adaptive_model:
        print("\nЛучшая модель: адаптивный шаг.")
    else:
        print("\nЛучшая модель: фиксированный шаг.")

    print("\n===== НЕСКОЛЬКО ПРЕДСКАЗАНИЙ НА TEST =====")
    test_predictions = best_model.predict(X_test)
    test_probabilities = best_model.predict_proba(X_test)

    for i in range(min(15, len(X_test))):
        print(
            f"Вход: {X_test[i].astype(int)} | "
            f"Истинный класс: {int(y_test[i][0])} | "
            f"Вероятность: {test_probabilities[i][0]:.6f} | "
            f"Предсказание: {int(test_predictions[i][0])}"
        )

    interactive_mode(best_model, N_INPUTS)


if __name__ == "__main__":
    main()