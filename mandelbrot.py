import matplotlib.pyplot as plt
import numpy as np


def recurrence_formula_x(x, y, a):
    return x**2 - y**2 + a


def recurrence_formula_y(x, y, b):
    return 2 * x * y + b


def is_convergence(a, b, max_iter=20, threshold=1000):
    x = np.zeros_like(a, dtype=np.float64)
    y = np.zeros_like(b, dtype=np.float64)

    diverged = np.zeros_like(a, dtype=bool)

    for _ in range(max_iter):
        x_new = recurrence_formula_x(x, y, a)
        y_new = recurrence_formula_y(x, y, b)

        newly_diverged = (np.abs(x_new) > threshold) | (np.abs(y_new) > threshold)

        diverged |= newly_diverged

        x = np.where(diverged, x, x_new)
        y = np.where(diverged, y, y_new)

    return ~diverged


a = np.arange(-1, 1.01, 0.001)
b = np.arange(-1, 1.01, 0.001)

A, B = np.meshgrid(a, b)

mask = is_convergence(A, B)
plt.figure(figsize=(6, 6))
plt.scatter(A[mask], B[mask], color="blue", s=2, label="Inside")
plt.legend()
plt.savefig("mandelbrot.png", dpi=300, format="png")
