import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [ 2,  6],
    [-2,  6],
    [ 2, -6],
    [-2, -6]
], dtype=float)

E = np.array([0, 1, 0, 0], dtype=float)

X1_MIN, X1_MAX = -3.0, 3.0
X2_MIN, X2_MAX = -7.0, 7.0

def step(u: float) -> int:
    return 1 if u >= 0 else 0

def net(x, w, b):
    return float(np.dot(w, x) + b)

def train_fixed_alpha(X, E, alpha=0.1, Ee=1e-4, max_epochs=5000):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    Es_hist = []
    epochs = 0

    while epochs < max_epochs:
        Es = 0.0
        for x, e in zip(X, E):
            y = net(x, w, b)
            err = e - y
            Es += err**2
            w += alpha * err * x
            b += alpha * err
        Es_hist.append(Es)
        epochs += 1
        if Es <= Ee:
            break
    return w, b, np.array(Es_hist), epochs

def train_adaptive_alpha(X, E, Ee=1e-4, max_epochs=5000):
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    Es_hist = []
    epochs = 0

    while epochs < max_epochs:
        Es = 0.0
        for x, e in zip(X, E):
            y = net(x, w, b)
            err = e - y
            Es += err**2

            denom = (np.dot(x, x) + 1.0)
            alpha_t = 1.0 / denom if denom != 0 else 0.0

            w += alpha_t * err * x
            b += alpha_t * err

        Es_hist.append(Es)
        epochs += 1
        if Es <= Ee:
            break
    return w, b, np.array(Es_hist), epochs

def plot_es(hist_fixed, hist_adapt):
    plt.figure()
    plt.plot(np.arange(1, len(hist_fixed) + 1), hist_fixed, label="fixed alpha")
    plt.plot(np.arange(1, len(hist_adapt) + 1), hist_adapt, label="adaptive alpha")
    plt.title("Es(p) learning curves")
    plt.xlabel("Epoch p")
    plt.ylabel("Es")
    plt.grid(True)
    plt.legend()

def plot_boundary_and_points(X, E, w, b, user_point=None):
    plt.figure()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)

    X0 = X[E == 0]
    X1c = X[E == 1]

    if len(X0) > 0:
        plt.scatter(X0[:, 0], X0[:, 1], marker="o", label="e=0")
    if len(X1c) > 0:
        plt.scatter(X1c[:, 0], X1c[:, 1], marker="s", label="e=1")

    w1, w2 = w[0], w[1]
    if abs(w2) >= 1e-12:
        x1_vals = np.linspace(X1_MIN, X1_MAX, 200)
        x2_vals = -(w1 * x1_vals + b) / w2
        plt.plot(x1_vals, x2_vals, label="boundary")
    elif abs(w1) >= 1e-12:
        x1_const = -b / w1
        plt.axvline(x=x1_const, label="boundary")

    if user_point is not None:
        u = net(np.array(user_point, dtype=float), w, b)
        cls = step(u)
        marker = "x" if cls == 0 else "*"
        plt.scatter([user_point[0]], [user_point[1]], marker=marker, s=150, label=f"user class={cls}")

    plt.xlim(X1_MIN, X1_MAX)
    plt.ylim(X2_MIN, X2_MAX)
    plt.legend()

def interactive_mode(w, b):
    plot_boundary_and_points(X, E, w, b)
    plt.show(block=False)

    while True:
        s = input("x1 x2 (or q): ").strip()
        if s.lower() in ("q", "quit", "exit"):
            break
        try:
            parts = s.replace(",", " ").split()
            if len(parts) != 2:
                print("Need 2 numbers: x1 x2")
                continue
            x1v = float(parts[0])
            x2v = float(parts[1])
        except:
            print("Bad input. Example: 2 -6")
            continue

        u = net(np.array([x1v, x2v], dtype=float), w, b)
        cls = step(u)
        print(f"class = {cls}")

        plot_boundary_and_points(X, E, w, b, user_point=[x1v, x2v])
        plt.show(block=False)

def main():
    Ee = 1e-4
    alpha_fixed = 0.1

    w_f, b_f, Es_f, ep_f = train_fixed_alpha(X, E, alpha=alpha_fixed, Ee=Ee)
    w_a, b_a, Es_a, ep_a = train_adaptive_alpha(X, E, Ee=Ee)

    print(f"Fixed alpha={alpha_fixed}: epochs={ep_f}, Es={Es_f[-1]:.6e}, w={w_f}, b={b_f:.6f}")
    print(f"Adaptive alpha: epochs={ep_a}, Es={Es_a[-1]:.6e}, w={w_a}, b={b_a:.6f}")
    if ep_a > 0:
        print(f"Speedup (fixed/adaptive) = {ep_f/ep_a:.2f}x")
    else:
        print("Adaptive finished immediately (unexpected).")

    plot_es(Es_f, Es_a)
    plt.show()

    plot_boundary_and_points(X, E, w_a, b_a)
    plt.title("Adaptive method: decision boundary")
    plt.show()

    interactive_mode(w_a, b_a)

if __name__ == "__main__":
    main()