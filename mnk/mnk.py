import numpy as np
import matplotlib.pyplot as plt
from typing import List


def Majewski_Tymoteusz_MNK(x: List, y: List, n: int, plot=True) -> List:
    """
    Aproksymacja średniokwadratowa metodą najmniejszych kwadratów == rozwiązanie układu równań A^T * A * a = A^T * y

    :param x: Lista punktów x
    :param y: Lista punktów y
    :param n: Stopień wielomianu
    :param plot: Czy rysować wykres?

    :returns: Lista współczynników wielomianu
    """

    if len(x) != len(y):
        raise ValueError("Długości x i y muszą być równe.")
    if n < 0:
        raise ValueError("Stopień wielomianu musi być nieujemny.")
    if n >= len(x):
        raise ValueError(
            "Stopień wielomianu musi być mniejszy od liczby punktów danych.")

    x = np.array(x)
    y = np.array(y)
    m = len(x)

    # Budowa macierzy A, gdzie kolumny to kolejne potęgi x
    A = []
    for i in range(m):
        row = [x[i]**j for j in range(n+1)]
        A.append(row)

    ATA = transpose(A) @ np.array(A)
    ATy = transpose(A) @ np.array(y)

    L = cholesky_decomposition(ATA)

    z = forward_substitution(L, ATy)
    a = backward_substitution(transpose(L), z)
    if plot:
        draw_plot(x, y, a)

    return a


def cholesky_decomposition(A):
    """
    Rozkład Cholesky'ego macierzy A na L * L^T, gdzie L jest macierzą dolnotrójkątną.

    :param A: Macierz do rozkładu
    :returns: Macierz dolnotrójkątna L rozkładu Cholesky'ego
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
    """
    Rozwiązanie układu równań L * y = b, gdzie L jest macierzą dolnotrójkątną.

    :param L: Macierz dolnotrójkątna
    :param b: Wektor prawej strony
    :returns: Wektor y rozwiązania
    """
    n = len(b)
    x = [0.0 for _ in range(n)]
    for i in range(n):
        s = sum(L[i][j] * x[j] for j in range(i))
        x[i] = (b[i] - s) / L[i][i]
    return x


def backward_substitution(U, b):
    """
    Rozwiązanie układu równań U * x = b, gdzie U jest macierzą górnotrójkątną.

    :param U: Macierz górnotrójkątna
    :param b: Wektor prawej strony
    :returns: Wektor x rozwiązania
    """
    n = len(b)
    x = [0.0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        s = sum(U[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - s) / U[i][i]
    return x


def transpose(M):
    """
    Transpozycja macierzy M.

    :param M: Macierz do transpozycji
    :returns: Transponowana macierz
    """
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]


def evaluate_polynomial(coeffs, x):
    """
    Oblicza wartość wielomianu o współczynnikach coeffs w punkcie x.

    :param coeffs: Lista współczynników wielomianu
    :param x: Punkt, w którym obliczamy wartość wielomianu
    :returns: Wartość wielomianu w punkcie x
    """
    return sum(coeffs[i] * x**i for i in range(len(coeffs)))


def draw_plot(x_data, y_data, coeffs):
    """
    Rysuje wykres punktów danych oraz wykres wielomianu aproksymacyjnego.
    
    :param x_data: Lista punktów x
    :param y_data: Lista punktów y
    :param coeffs: Lista współczynników wielomianu    
    """
    x_min = min(x_data)
    x_max = max(x_data)
    xs = [x_min + i * (x_max - x_min) / 500 for i in range(501)]
    ys = [evaluate_polynomial(coeffs, x) for x in xs]

    plt.scatter(x_data, y_data, color='blue', label='Znane punkty')
    plt.plot(xs, ys, color='grey', label='Wielomian aproksymacyjny')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Aproksymacja metodą najmniejszych kwadratów")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    x = [1, 2, 3, 4, 5]
    y = [2.2, 2.8, 3.6, 4.5, 5.1]
    n = 4

    coeffs = Majewski_Tymoteusz_MNK(x, y, n)
    print("Współczynniki wielomianu:", coeffs)

if __name__ == "__main__":
    main()