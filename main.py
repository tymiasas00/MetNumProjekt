import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

def Majewski_Tymoteusz_MNK(x: List, y: List, n: int) -> List:
    """
    Aproksymacja średniokwadratowa metodą najmniejszych kwadratów
    z użyciem rozkładu Cholesky'ego – własna implementacja + NumPy tylko do mnożenia macierzy.
    """
    x = list(x)
    y = list(y)
    m = len(x)

    # Tworzenie macierzy A (Vandermonde)
    A = []
    for i in range(m):
        row = [x[i]**j for j in range(n+1)]
        A.append(row)


    ATA = np.array(transpose(A)) @ np.array(A)
    ATy = np.array(transpose(A)) @ np.array(y)

    # Rozkład Cholesky'ego (własny)
    L = cholesky_decomposition(ATA.tolist())

    # Rozwiązanie układów
    z = forward_substitution(L, ATy.tolist())
    a = backward_substitution(transpose(L), z)

    # Rysowanie wykresu
    draw_plot(x, y, a)

    return a


def cholesky_decomposition(A):
    """
    Rozkład Cholesky'ego macierzy A na L * L^T, gdzie L jest macierzą dolnotrójkątną.
    """
    n = len(A)
    L = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(i+1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                L[i][j] = (A[i][i] - s) ** 0.5
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def forward_substitution(L, b):
    n = len(b)
    x = [0.0 for _ in range(n)]
    for i in range(n):
        s = sum(L[i][j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


def backward_substitution(U, b):
    n = len(b)
    x = [0.0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - s) / U[i][i]
    return x


def transpose(M):
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def evaluate_polynomial(coeffs, x):
    return sum(coeffs[i] * x**i for i in range(len(coeffs)))


def draw_plot(x_data, y_data, coeffs):
    x_min = min(x_data)
    x_max = max(x_data)
    xs = [x_min + i * (x_max - x_min) / 500 for i in range(501)]
    ys = [evaluate_polynomial(coeffs, x) for x in xs]

    plt.scatter(x_data, y_data, color='red', label='Punkty dane')
    plt.plot(xs, ys, color='blue', label='Wielomian aproksymacyjny')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Aproksymacja metodą najmniejszych kwadratów")
    plt.grid(True)
    plt.legend()
    plt.show()

x = [0, 1, 2, 3, 4]
y = [1, 2, 0.9, 3, 7]
n = 2

a = Majewski_Tymoteusz_MNK(x, y, n)
print("Współczynniki wielomianu:", a)