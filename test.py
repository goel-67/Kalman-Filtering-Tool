import numpy as np
from Kalman import (
    kalman,
)  # Make sure to replace `your_module_name` with the actual name of your Python module


# Define the Kalman filter test function
def test_kalman_filter():
    # Initialize variables based on the MATLAB example
    k = 0
    Z = np.array([0.0101])  # Measurement values
    u = np.array([0.0101, 0.1188])  # Initial state vector (mean)
    X = np.array(
        [[0.01071225, 0.017495523], [0.017495523, 2.04175521]]  # Covariance matrix (X)
    )
    V = np.array([0, 0])  # Gaussian influence diagram conditional variances
    R = np.array([0.01])  # Measurement noise covariance matrix
    H = np.array([[1, 0]])  # Measurement matrix
    Phi = np.array([[1.0191, 0.0099], [-0.2474, 0.9994]])  # State transition matrix
    gamma = np.array([[1, 0], [0, 1]])  # Process noise matrix
    Qk = np.array([[0.002, 0.002], [0.002, 0.438]])  # Process noise covariance matrix
    Form = 1  # To convert the result back to covariance form

    # Run the Kalman filter
    u_updated, B_updated, V_updated = kalman(k, Z, u, X, V, R, H, Phi, gamma, Qk, Form)

    # Output the results
    print("Updated state (u):")
    print(u_updated)

    print("\nUpdated covariance matrix (B):")
    print(B_updated)

    print("\nUpdated conditional variances (V):")
    print(V_updated)


# Run the test
if __name__ == "__main__":
    test_kalman_filter()
