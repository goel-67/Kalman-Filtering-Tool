import numpy as np
from COVtoINF import cov_to_inf
from Evidence import evidence
from INFtoCOV import inf_to_cov


def mupdate(k, Z, u, B_or_sigma, V, R, H):
    """
    Measurement update for measurement Z(k) in the Kalman filter.

    Parameters:
    k (int): Desired discrete time.
    Z (numpy.ndarray): A p x 1 vector of measurement values at discrete time k.
    u (numpy.ndarray): An n x 1 vector representing the mean of the state at discrete time k.
    B_or_sigma (numpy.ndarray): If k = 0, the covariance matrix. If k ≠ 0, the Gaussian influence diagram arc coefficients.
    V (numpy.ndarray): If k ≠ 0, a vector of Gaussian influence diagram conditional variances.
    R (numpy.ndarray): The diagonal measurement noise covariance matrix R.
    H (numpy.ndarray): The measurement matrix at discrete time k.

    Returns:
    u (numpy.ndarray): The updated mean vector of the state at discrete time k+1.
    V (numpy.ndarray): The updated vector of conditional variances of the state at discrete time k+1.
    B (numpy.ndarray): The updated matrix of Gaussian influence diagram arc coefficients at discrete time k+1.
    """

    # Get dimensions
    n = V.shape[0]
    p = Z.shape[0]

    if k == 0:
        # Convert covariance matrix to influence diagram form
        B, V, _ = cov_to_inf(B_or_sigma, n)

    # Setup for Evidence update
    X1 = Z.copy()  # Evidence vector
    n0 = n
    n1 = p
    n2 = 0

    u_new = np.concatenate([u, H @ u])
    du = np.zeros(n + p)

    V_new = np.concatenate([V, np.diag(R)])
    Opn = np.zeros((p, n))
    Opp = np.zeros((p, p))

    B_new_top = np.hstack((B, H.T))
    B_new_bottom = np.hstack((Opn, Opp))
    B_new = np.vstack((B_new_top, B_new_bottom))

    # Perform Evidence update
    u, B, V = evidence(u_new, B_new, V_new, X1, n0, n1, n2, du)

    # Extract the relevant parts for the updated state
    u = u[:n]
    B = B[:n, :n]
    V = V[:n]

    return u, V, B

# Example test case
k = 0
Z = np.array([[3], [4]])  # Measurement vector
u = np.array([[1], [2]])  # Initial state vector
B_or_sigma = np.array([[4, 1], [1, 9]])  # Covariance matrix
V = np.array([[0], [0]])  # Conditional variances (influence diagram)
R = np.array([[1, 0], [0, 4]])  # Measurement noise covariance matrix
H = np.array([[0, 2], [3, 0]])  # Measurement matrix

# Perform the measurement update
u, V, B = mupdate(k, Z, u, B_or_sigma, V, R, H)

# Convert influence diagram form back to covariance form
n = u.shape  # Shape of the state vector
X = inf_to_cov(V, B, n[0])

# Display the results
np.set_printoptions(precision=5)
print("Updated State (u):")
print(u)
print("Updated Covariance Matrix (X):")
print(X)