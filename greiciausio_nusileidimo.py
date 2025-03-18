import numpy as np
import matplotlib.pyplot as plt

# be atvaizdavimo

# ===== Tikslo funkcija ir jos gradientas =====

def f(X):
    x, y = X
    return - (x * y * (1 - x - y)) / 8


def grad_f(X):
    x, y = X
    dfdx = - (y * (1 - 2 * x - y)) / 8
    dfdy = - (x * (1 - x - 2 * y)) / 8
    return np.array([dfdx, dfdy])


# ----- ternary search -----
def line_search_1d(f, X, grad, gamma_max=1.0, tol=1e-8, max_steps=100):
    """
    Ternary search – randame γ ∈ [0, gamma_max], kuris minimizuoja
    φ(γ) = f(X - γ * grad).

    Grąžina: optimalia γ reikšmę.
    """
    left, right = 0.0, gamma_max
    for _ in range(max_steps):
        if abs(right - left) < tol:
            break
        m1 = left + (right - left) / 3
        m2 = right - (right - left) / 3
        phi_m1 = f(X - m1 * grad)
        phi_m2 = f(X - m2 * grad)
        if phi_m1 > phi_m2:
            left = m1
        else:
            right = m2
    return (left + right) / 2


# ===== Greičiausiojo nusileidimo metodas =====

def steepest_descent_slides(f, grad_f, X0, eps=1e-8, gamma_max=1.0, max_iter=1000):
    """
    Greičiausias nusileidimas:

      1) i = 0
      2) X_{i+1} = X_i - γ * grad_f(X_i), kur
         γ = argmin_{γ ≥ 0} f(X_i - γ * grad_f(X_i))
      3) Jei ||grad_f(X_{i+1})|| < eps, sustojame; kitaip i = i + 1 ir kartojame 2)
    """
    X = np.array(X0, dtype=float)
    traj = [X.copy()]
    grad_calls = 0
    i = 0
    while i < max_iter:
        g = grad_f(X)
        grad_calls += 1

        #  optimalųs γ, naudojant ternary search
        best_gamma = line_search_1d(f, X, g, gamma_max=gamma_max)

        # Naujas taškas
        X_next = X - best_gamma * g

        # Patikriname sustojimo sąlygą: gradiento norma naujame taške
        if np.linalg.norm(grad_f(X_next)) < eps:
            traj.append(X_next.copy())
            print(f"Greiciausias nusileidimas: konvergencija pasiekta per {i + 1} iteracijų.")
            return X_next, i + 1, grad_calls, traj

        X = X_next
        traj.append(X.copy())
        i += 1

    print("Greiciausias nusileidimas: nepasiekta konvergencija per max_iter.")
    return X, i, grad_calls, traj


# ===== Pagrindinė dalis: išbandome metodą su pradiniu tašku =====

if __name__ == "__main__":
    # Pasirenkame pradines reikšmes
    X0 = [0.0, 0.0]
    X1 = [1.0, 1.0]
    Xm = [0.7, 0.8]  # a=7, b=8

    # Parametrai
    eps = 1e-8
    gamma_max = 1.0
    max_iter = 1000

    for label, X_start in zip(["X0", "X1", "Xm"], [X0, X1, Xm]):
        sol, iters, grad_calls, traj = steepest_descent_slides(f, grad_f, X_start,
                                                               eps=eps,
                                                               gamma_max=gamma_max,
                                                               max_iter=max_iter)
        print("-" * 50)
        print(f"Pradinis taškas {label} = {X_start}")
        print(f"  Rastas sprendinys  = [{sol[0]:.8f}, {sol[1]:.8f}]")
        print(f"  f(sprendinys)       = {f(sol):.8f}")
        print(f"  Iteracijų skaičius   = {iters}")
        print(f"  Gradiento kvietimų  = {grad_calls}")
        print("-" * 50)


