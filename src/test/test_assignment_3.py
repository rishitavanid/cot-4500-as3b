import numpy as np
import pytest
from src.main.assignment_3 import (
    gaussian_elimination,
    lu_decomposition_doolittle,
    is_diagonally_dominant,
    is_positive_definite
)

# Test for Gaussian Elimination function
def test_gaussian_elimination():
    # Augmented matrix representing a system of equations
    A1 = np.array([
        [2, -1, 1, 6],
        [1, 3, 1, 0],
        [-1, 5, 4, -3]
    ])
    expected = np.array([2, -1, 1])  # Expected integer solution to the system
    result = gaussian_elimination(A1.copy())  # Perform Gaussian Elimination
    assert np.array_equal(result, expected), f"Expected {expected}, but got {result}"

# Test for LU decomposition using Doolittle’s method
def test_lu_decomposition_doolittle():
    # Input square matrix for LU decomposition
    A2 = np.array([
        [1, 1, 0, 3],
        [2, 1, -1, 1],
        [3, -1, -1, 2],
        [-1, 2, 3, -1]
    ])

    # Expected L (Lower triangular) matrix
    L_expected = np.array([
        [1, 0, 0, 0],
        [2, 1, 0, 0],
        [3, 4, 1, 0],
        [-1, -3, 0, 1]
    ], dtype=float)

    # Expected U (Upper triangular) matrix
    U_expected = np.array([
        [1, 1, 0, 3],
        [0, -1, -1, -5],
        [0, 0, 3, 13],
        [0, 0, 0, -13]
    ], dtype=float)

    det_expected = 39.0  # Expected determinant value

    # Run LU decomposition
    L, U, det = lu_decomposition_doolittle(A2.copy())

    # Check if computed L and U are close to expected values
    assert np.allclose(L, L_expected), f"L matrix incorrect. Got: {L}"
    assert np.allclose(U, U_expected), f"U matrix incorrect. Got: {U}"
    # Check if determinant matches expected
    assert np.isclose(det, det_expected), f"Determinant incorrect. Got: {det}"

# Test for checking if a matrix is diagonally dominant
def test_diagonally_dominant():
    A3 = np.array([
        [9, 0, 5, 2, 1],
        [3, 9, 1, 2, 1],
        [0, 1, 7, 2, 3],
        [4, 2, 3, 12, 2],
        [3, 2, 4, 0, 8]
    ])
    # This matrix is NOT diagonally dominant, so the function should return False
    assert is_diagonally_dominant(A3) == False

# Test for checking if a matrix is symmetric and positive definite
def test_positive_definite():
    A4 = np.array([
        [2, 2, 1],
        [2, 3, 0],
        [1, 0, 2]
    ])
    # Verify the matrix is symmetric
    assert np.allclose(A4, A4.T), "Matrix is not symmetric"
    # Verify the matrix is positive definite
    assert is_positive_definite(A4), "Matrix is not positive definite"
