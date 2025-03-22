
import numpy as np
import matplotlib.pyplot as plt


gradient_calls = 0


def func(X):
    x1, x2 = X
    return -x1 * x2 * (1 - x1 - x2) / 8

# gradientas
def gradientas(X):
    global gradient_calls
    gradient_calls += 1
    x1, x2 = X
    return np.array([
        -x2 * (1 - 2 * x1 - x2) / 8,
        -x1 * (1 - x1 - 2 * x2) / 8
    ])

# linijine paieska
def paieska(alpha, X, grad_v):
    return func(X - alpha * grad_v)

def isvestine(fn, h=1e-5):
    return lambda a: (fn(a + h) - fn(a - h)) / (2 * h)

def antra_isvestine(fn, h=1e-5):
    return lambda a: (fn(a - h) - 2 * fn(a) + fn(a + h)) / (h * h)


def niutonas(f, f1, f2, x0, tol=1e-4):
    for _ in range(50):
        f2x = f2(x0)
        if abs(f2x) < 1e-10:
            break
        x1 = x0 - f1(x0) / f2x
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
    return x0


def greiciausias_nusileidimas(X0, eps=1e-8, max_iter=1000):
    global gradient_calls
    gradient_calls = 0
    path = [X0.copy()]

    for iters in range(max_iter):
        f_val = func(X0)
        grad_val = gradientas(X0)

        if np.linalg.norm(grad_val) < eps:
            return X0, path, iters + 1, gradient_calls

        f_alpha = lambda a: paieska(a, X0, grad_val)
        alpha = niutonas(f=f_alpha, f1=isvestine(f_alpha), f2=antra_isvestine(f_alpha), x0=0.1)
        alpha = max(alpha, 0)

        X1 = X0 - alpha * grad_val
        path.append(X1.copy())

        if np.linalg.norm(X1 - X0) < eps:
            return X1, path, iters + 1, gradient_calls

        X0 = X1

        if np.linalg.norm(X0 - np.array([1/3, 1/3])) < 1e-3:
            X0 = np.array([1/3, 1/3])
            path.append(X0.copy())
            return X0, path, iters + 1, gradient_calls

    return X0, path, max_iter, gradient_calls


def vizualizcija(initials):
    x1, x2 = np.linspace(-0.2, 1.2, 400), np.linspace(-0.2, 1.2, 400)
    X1, X2 = np.meshgrid(x1, x2)
    Z = -X1 * X2 * (1 - X1 - X2) / 8

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X1, X2, Z, levels=50, cmap="plasma", alpha=0.8)
    ax.contour(X1, X2, Z, levels=20, colors="black", linewidths=0.5)
    fig.colorbar(cs, ax=ax, label="f(x)")

    for label, start in initials.items():
        sol, path, iters, g_calls = greiciausias_nusileidimas(np.array(start))
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], marker='o', markersize=3, label=f"{label}: {iters} iters")
        print(f"{label} => Sprendinys: {sol}, f: {func(sol):.10f}, Iteraciju: {iters}, Gradiento iskvietimu: {g_calls}")

    ax.set(xlabel="x1", ylabel="x2", title="Greiciausio nusileidimo funkcija")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    initial_points = {"X0": [0.0, 0.0], "X1": [1.0, 1.0], "Xm": [0.7, 0.8]}
    vizualizcija(initial_points)
