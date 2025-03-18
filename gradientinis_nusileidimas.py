import numpy as np
import matplotlib.pyplot as plt


# Funkcija ir gradientas
def f(X):
    x, y = X
    return -(x * y * (1 - x - y)) / 8


def grad_f(X):
    x, y = X
    dfdx = -(y * (1 - 2 * x - y)) / 8
    dfdy = -(x * (1 - x - 2 * y)) / 8
    return np.array([dfdx, dfdy])


# Gradientinis nusileidimas
def gradient_descent(f, grad_f, X0, gamma, eps, max_iter=10_000):
    i = 0
    X = np.array(X0, dtype=float)
    path = [X.copy()]

    while i < max_iter:
        X_next = X - gamma * grad_f(X)
        path.append(X_next.copy())
        if np.linalg.norm(grad_f(X_next)) < eps:
            return X_next, i + 1, np.array(path)
        X = X_next
        i += 1

    return X, i, np.array(path)


# Pradiniai taškai ir parametrai
initial_points = {"X0": [0.0, 0.0], "X1": [1.0, 1.0], "Xm": [0.7, 0.8]}
gamma = 0.1
eps = 1e-8
max_iter = 10_000


# Vykdymas su rezultatais konsolėje
for label, X_start in initial_points.items():
    sol, iters, path = gradient_descent(f, grad_f, X_start, gamma, eps, max_iter)
    plt.plot(path[:, 0], path[:, 1], '-o', markersize=3, label=f'{label}')

    # Spausdinami rezultatai į konsolę
    print(f"Pradinis taškas {label} = {X_start}")
    print(f"  Rastas sprendinys = [{sol[0]:.8f}, {sol[1]:.8f}]")
    print(f"  f(sprendinys)     = {f(sol):.8f}")
    print(f"  Iteracijų skaičius= {iters}")
    print('-' * 50)

