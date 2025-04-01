# COT 4500 – Assignment 3B: Numerical Calculus

This assignment covers Chapters 5 and 6, focusing on matrix operations such as **Gaussian Elimination**, **LU Decomposition**, **Diagonal Dominance**, and **Positive Definiteness**.

The goal is to apply numerical methods using Python and NumPy to solve systems of linear equations and assess matrix properties without relying on external libraries like SciPy.

---

## Project Structure

```
cot-4500-as3b/
│
├── src/
│   ├── main/
│   │   ├── __init__.py
│   │   └── assignment_3.py        # Main implementation of all numerical methods
│   │
│   └── test/
│       ├── __init__.py
│       └── test_assignment_3.py   # Unit tests for each question
│
├── requirements.txt               # Lists required Python packages
├── README.md                      # This file
```

---

## How to Run the Program

1. Make sure you have Python 3.8+ and pip installed.

2. (Optional but recommended) Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Run the assignment code to see all four questions executed:
```bash
python src/main/assignment_3.py
```

---

## Expected Output

```
Question 1:
[ 2 -1  1]

Question 2:
Matrix Determinant: 39.0
L Matrix:
 [[ 1.  0.  0.  0.]
  [ 2.  1.  0.  0.]
  [ 3.  4.  1.  0.]
  [-1. -3.  0.  1.]]
U Matrix:
 [[  1.   1.   0.   3.]
  [  0.  -1.  -1.  -5.]
  [  0.   0.   3.  13.]
  [  0.   0.   0. -13.]]

Question 3:
Is diagonally dominant? False

Question 4:
Is symmetric? True
Is positive definite? True
```

---

## How to Run the Tests

Unit tests for all four questions are included using `pytest`.

To run the tests:

```bash
pytest src/test/test_assignment_3.py
```

If you want verbose output:
```bash
pytest -v src/test/test_assignment_3.py
```

---

## Requirements

Only `numpy` and `pytest` are required:

```
numpy>=1.24.0
pytest>=7.0.0
```

---

## Restrictions

- Do **not** use external libraries like `scipy` for LU decomposition or matrix checking.
- Everything must be implemented manually using `numpy`.

---
