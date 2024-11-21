import numpy as np
from Mupdate import mupdate
from Tupdate import tupdate
from INFtoCOV import inf_to_cov

def kalman(k, Z, u, X, V, R, H, Phi, gamma, Qk, Form):
    """
    Apply Kalman filter at time k.

    Parameters:
    k (int): Desired discrete time.
    Z (numpy.ndarray): Measurement values at discrete time k.
    u (numpy.ndarray): An n x 1 vector representing the mean of the state at time k.
    X (numpy.ndarray): If k=0, the covariance matrix of state at time k. If k≠0, the matrix of Gaussian influence diagram arc coefficients.
    V (numpy.ndarray): If k≠0, an n x 1 vector of Gaussian influence diagram conditional variances. Ignored if k=0.
    R (numpy.ndarray): The measurement noise covariance matrix R.
    H (numpy.ndarray): The measurement matrix at discrete time k.
    Phi (numpy.ndarray): The state transition matrix at time k.
    gamma (numpy.ndarray): The process noise matrix at time k.
    Qk (numpy.ndarray): The process noise covariance matrix.
    Form (int): Determines the output form (0 for ID form, 1 for covariance form).

    Returns:
    u (numpy.ndarray): The updated mean vector.
    B (numpy.ndarray): The updated state covariance matrix or influence diagram form matrix.
    V (numpy.ndarray): The updated vector of conditional variances.
    """

    # Get dimensions
    domain = X.shape[0]
    p = Z.shape[0]
    
    # Perform measurement update
    u, V, B = mupdate(k, Z, u, X, V, R, H)
    u_new = u[:domain]
    V_new = V[:domain]
    B_new = B[:domain, :domain]

    # Perform time update
    u, B, V = tupdate(u_new, B_new, V_new, Phi, gamma, Qk)

    # Convert back to covariance form if required
    if Form == 1:
        B = inf_to_cov(V, B, domain)

    return u, B, V

# Testing the Kalman filter function with MATLAB script values

# Initial test values (based on the MATLAB script)
k = 0
Z = np.array([[0.0101]])  # Measurement values
u = np.array([[0.0101], [0.1188]])  # Initial state mean vector
X = np.array([[0.01071225, 0.017495523], [0.017495523, 2.04175521]])  # Covariance matrix
V = np.array([[0], [0]])  # Conditional variances
R = np.array([[0.01]])  # Measurement noise covariance matrix
H = np.array([[1, 0]])  # Measurement matrix
Phi = np.array([[1.0191, 0.0099], [-0.2474, 0.9994]])  # State transition matrix
gamma = np.array([[1, 0], [0, 1]])  # Process noise matrix
Qk = np.array([[0.002, 0.002], [0.002, 0.438]])  # Process noise covariance matrix
Form = 1  # Output form, 1 for covariance form

'''k = 0  # Initial time step
Z = np.array([[502.55], [-0.9316]])  # Measurement values (p * 1)
u = np.array([[400], [0], [0], [-300], [0], [0]])  # Initial state mean vector (n * 1)
X = np.array([
    [1125, 750, 250, 0, 0, 0],
    [750, 1000, 500, 0, 0, 0],
    [250, 500, 500, 0, 0, 0],
    [0, 0, 0, 1125, 750, 250],
    [0, 0, 0, 750, 1000, 500],
    [0, 0, 0, 250, 500, 500]
])  # Covariance matrix (n * n)
V = np.array([[0], [0], [0], [0], [0], [0]])  # Conditional variances (n * 1)
R = np.array([[25, 0], [0, 0.0087**2]])  # Measurement noise covariance matrix (p * p)
H = np.array([
    [0.8, 0, 0, -0.6, 0, 0],
    [0.0012, 0, 0, 0.0016, 0, 0]
])  # Measurement matrix (p * n)
Phi = np.array([
    [1, 1, 0.5, 0, 0, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0.5],
    [0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1]
])  # State transition matrix (n * n)
gamma = np.eye(6)  # Process noise matrix (n * n)
Qk = np.array([
    [0.25, 0.5, 0.5, 0, 0, 0],
    [0.5, 1, 1, 0, 0, 0],
    [0.5, 1, 1, 0, 0, 0],
    [0, 0, 0, 0.25, 0.5, 0.5],
    [0, 0, 0, 0.5, 1, 1],
    [0, 0, 0, 0.5, 1, 1]
])  # Process noise covariance matrix (n * n)

Qk = Qk * 0.2**2

Form = 1  # Output form, 1 for covariance form'''

# Run the Kalman filter
u_updated, B_updated, V_updated = kalman(k, Z, u, X, V, R, H, Phi, gamma, Qk, Form)

# Display the results
np.set_printoptions(precision=5)
print("Updated State (u):")
print(u_updated)
print("Updated Covariance Matrix (B or X):")
print(B_updated)
print("Updated Conditional Variances (V):")
print(V_updated)
