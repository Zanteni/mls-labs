"""
Correctness tests — run against the student's extracted functions.

Assumes `student_solution.py` (produced from the student's notebook via
`jupyter nbconvert --to script`) is importable from this directory.

Usage:
    jupyter nbconvert --to script student_lab_template.ipynb --output student_solution
    pytest grading/test_student_lab.py --junitxml=results.xml
"""

import sys
import os
import numpy as np
import pytest

# Make the converted student script importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from student_solution import compute_cost, gradient_descent  # noqa: E402


# --- Fixtures: small, hand-computable toy data -----------------------------

@pytest.fixture
def toy_data():
    # X includes bias column. y chosen so cost is hand-verifiable.
    X = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    y = np.array([2.0, 3.0, 5.0])
    theta = np.array([1.0, 1.0])
    return X, y, theta


# --- compute_cost -----------------------------------------------------------

def test_compute_cost_returns_float(toy_data):
    X, y, theta = toy_data
    cost = compute_cost(X, y, theta)
    assert isinstance(cost, (int, float, np.floating))


def test_compute_cost_correct_value(toy_data):
    X, y, theta = toy_data
    # predictions = [2, 3, 4]; errors = [0, 0, -1]; sum(sq)=1; m=3
    expected = 1 / (2 * 3)
    cost = compute_cost(X, y, theta)
    assert abs(cost - expected) < 1e-6, f"Expected {expected}, got {cost}"


def test_compute_cost_zero_when_perfect_fit():
    X = np.array([[1.0, 1.0], [1.0, 2.0]])
    y = np.array([2.0, 3.0])
    theta = np.array([1.0, 1.0])  # perfect fit: pred = [2, 3]
    cost = compute_cost(X, y, theta)
    assert abs(cost) < 1e-9


# --- gradient_descent --------------------------------------------------------

def test_gradient_descent_output_shapes(toy_data):
    X, y, theta = toy_data
    theta_final, cost_history = gradient_descent(X, y, theta, lr=0.01, n_iters=50)
    assert theta_final.shape == theta.shape
    assert len(cost_history) == 50


def test_gradient_descent_cost_decreases(toy_data):
    X, y, theta = toy_data
    theta_final, cost_history = gradient_descent(X, y, np.zeros(2), lr=0.1, n_iters=200)
    assert cost_history[-1] < cost_history[0], "Cost should decrease over training"


def test_gradient_descent_converges_to_reasonable_fit(toy_data):
    X, y, _ = toy_data
    theta_final, cost_history = gradient_descent(X, y, np.zeros(2), lr=0.1, n_iters=2000)
    assert cost_history[-1] < 0.5, "Cost did not converge to a reasonably low value"


def test_gradient_descent_does_not_mutate_input_theta(toy_data):
    X, y, theta = toy_data
    theta_copy = theta.copy()
    gradient_descent(X, y, theta, lr=0.01, n_iters=10)
    assert np.allclose(theta, theta_copy), "gradient_descent should not mutate the input theta in place"
