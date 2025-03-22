import numpy as np
import matplotlib.pyplot as plt


tiksloFunkcija_calls = 0

def project_to_domain(x):
    x = np.maximum(x, 0)
    total = x[0] + x[1]
    if total > 1:
        x = x / total
    return x

def tiksloFunkcija(X):
    global tiksloFunkcija_calls
    tiksloFunkcija_calls += 1
    x1, x2 = X
    if x1 < 0 or x2 < 0 or (x1 + x2 > 1):
        return np.inf
    return -(x1 * x2 * (1 - x1 - x2)) / 8

def deformuojamas_simpleksas(f, x0, alpha=1.0, beta=0.5, gamma=2.0, tol=1e-6, max_iter=1000):
    x0 = project_to_domain(x0)

    n = len(x0)
    simplex = np.array([project_to_domain(x0 + np.eye(n)[i] * 0.1) for i in range(n)] + [x0])
    values = np.array([f(x) for x in simplex])
    simplex_history = [simplex.copy()]

    for _ in range(max_iter):
        order = np.argsort(values)
        simplex = simplex[order]
        values = values[order]

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = project_to_domain(centroid + alpha * (centroid - simplex[-1]))
        f_reflected = f(reflected)

        if f_reflected < values[0]:
            expanded = project_to_domain(centroid + gamma * (reflected - centroid))
            f_expanded = f(expanded)
            if f_expanded < f_reflected:
                simplex[-1] = expanded
                values[-1] = f_expanded
            else:
                simplex[-1] = reflected
                values[-1] = f_reflected
        elif f_reflected < values[-2]:
            simplex[-1] = reflected
            values[-1] = f_reflected
        else:
            contracted = project_to_domain(centroid + beta * (simplex[-1] - centroid))
            f_contracted = f(contracted)
            if f_contracted < values[-1]:
                simplex[-1] = contracted
                values[-1] = f_contracted
            else:
                for i in range(1, n + 1):
                    reduced = project_to_domain(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                    simplex[i] = reduced
                    values[i] = f(reduced)

        simplex_history.append(simplex.copy())

        max_distance = np.max(np.linalg.norm(simplex - simplex[0], axis=1))
        if max_distance < tol:
            break

    best_index = np.argmin(values)
    return simplex[best_index], values[best_index], simplex_history

def visualize_simplex(history, label):
    for simplex in history:
        plt.plot(simplex[:, 0], simplex[:, 1], 'o-', markersize=3, alpha=0.3, label=label if simplex is history[0] else "")

def main():
    global tiksloFunkcija_calls
    initial_points = {"X0": [0, 0], "X1": [1, 1], "Xm": [0.1, 0.2]}

    x = np.linspace(-0.2, 1.2, 300)
    y = np.linspace(-0.2, 1.2, 300)
    X, Y = np.meshgrid(x, y)
    Z = -(X * Y * (1 - X - Y)) / 8

    plt.figure(figsize=(10, 8))

    contour = plt.contour(X, Y, Z, levels=30, cmap='viridis')
    plt.clabel(contour, inline=True, fontsize=8)

    for label, start in initial_points.items():
        tiksloFunkcija_calls = 0
        projected_start = project_to_domain(np.array(start))
        optimal_x, optimal_value, history = deformuojamas_simpleksas(tiksloFunkcija, projected_start, tol=1e-6)

        print(f"{label}: sprendinys = {optimal_x}, funkcijos reiksme = {optimal_value:.10f}")
        print(f"  Funkcijos iskvietimai: {tiksloFunkcija_calls}")
        print("-" * 50)

        visualize_simplex(history, label)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.ylim(-0.2, 0.6)
    plt.tight_layout()
    plt.show()

main()
