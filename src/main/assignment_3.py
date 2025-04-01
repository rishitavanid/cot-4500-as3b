import numpy as np

# Performs Gaussian Elimination with partial pivoting
def gaussian_elimination(A):
    A = A.astype(float)  # Ensure matrix has float type for division
    n = len(A)

    # Forward Elimination
    for i in range(n):
        # Find pivot row with max absolute value in current column and swap
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]

        # Normalize the pivot row
        A[i] = A[i] / A[i][i]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            A[j] = A[j] - A[j][i] * A[i]

    # Back Substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = A[i, -1] - np.sum(A[i, i + 1:n] * x[i + 1:n])

    return x.astype(int)  # Return integer solution

# Performs LU Decomposition using Doolittle’s method
def lu_decomposition_doolittle(A):
    A = A.astype(float)
    n = A.shape[0]
    L = np.eye(n)      # Lower triangular matrix initialized as identity
    U = A.copy()       # Upper triangular matrix

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]  # Multiplier for row elimination
            L[j, i] = factor           # Store multiplier in L
            U[j] -= factor * U[i]     # Subtract from lower rows

    det = np.prod(np.diag(U))         # Determinant from product of U diagonal
    return L, U, det

# Checks if a matrix is diagonally dominant
def is_diagonally_dominant(matrix):
    for i in range(len(matrix)):
        diag = abs(matrix[i, i])  # Diagonal element
        off_diag_sum = sum(abs(matrix[i, j]) for j in range(len(matrix)) if j != i)  # Sum of other elements
        if diag < off_diag_sum:
            return False
    return True

# Checks if a matrix is positive definite using leading principal minors
def is_positive_definite(matrix):
    if not np.allclose(matrix, matrix.T):  # Must be symmetric
        return False
    for k in range(1, matrix.shape[0] + 1):
        minor = matrix[:k, :k]  # Leading principal minor
        if np.linalg.det(minor) <= 0:
            return False
    return True


# === Executing All Questions ===

# Question 1: Solve a system using Gaussian Elimination
A1 = np.array([
    [2, -1, 1, 6],
    [1, 3, 1, 0],
    [-1, 5, 4, -3]
])
solution_q1 = gaussian_elimination(A1)
print("Question 1:")
print(solution_q1)

# Question 2: LU Decomposition of a 4x4 matrix
A2 = np.array([
    [1, 1, 0, 3],
    [2, 1, -1, 1],
    [3, -1, -1, 2],
    [-1, 2, 3, -1]
])
L, U, det = lu_decomposition_doolittle(A2)
print("\nQuestion 2:")
print("Matrix Determinant:", det)
print("L Matrix:\n", L)
print("U Matrix:\n", U)

# Question 3: Check if a matrix is diagonally dominant
A3 = np.array([
    [9, 0, 5, 2, 1],
    [3, 9, 1, 2, 1],
    [0, 1, 7, 2, 3],
    [4, 2, 3, 12, 2],
    [3, 2, 4, 0, 8]
])
result_q3 = is_diagonally_dominant(A3)
print("\nQuestion 3:")
print("Is diagonally dominant?", result_q3)

# Question 4: Check if a matrix is symmetric and positive definite
A4 = np.array([
    [2, 2, 1],
    [2, 3, 0],
    [1, 0, 2]
], dtype=float)
symmetry = np.allclose(A4, A4.T)
pos_def = is_positive_definite(A4)
print("\nQuestion 4:")
print("Is symmetric?", symmetry)
print("Is positive definite?", pos_def)
