import numpy as np
from COVtoINF import cov_to_inf
from Tupdate import tupdate
from INFtoCOV import inf_to_cov

class trackingKF:
    def __init__(self, F, H, state, state_covariance, process_noise):
        self.StateTransitionModel = F
        self.MeasurementModel = H
        self.State = state
        self.StateCovariance = state_covariance
        self.ProcessNoise = process_noise

def id_predict(filter):
    """
    Performs the Time Update (prediction) portion of the Kalman Filter for object-oriented programming.

    Parameters:
    filter (object): A trackingKF object that contains State, StateCovariance, MeasurementModel, StateTransitionModel, and ProcessNoise.

    Returns:
    xpred (numpy.ndarray): The predicted state.
    Ppred (numpy.ndarray): The predicted state estimation error covariance.
    """

    # Extract filter parameters
    u = filter.State
    B_old = filter.StateCovariance
    H = filter.MeasurementModel
    F = filter.StateTransitionModel
    Qk = filter.ProcessNoise
    
    M = Qk.shape[0]
    I = np.eye(M)

    # Convert covariance to influence diagram form
    B_old, V_old, Precision = cov_to_inf(B_old, M)

    # Perform time update
    u, B, V = tupdate(u, B_old, V_old, F, I, Qk)

    # Convert influence diagram back to covariance form
    Ppred = inf_to_cov(V, B, M)

    xpred = u

    return xpred, Ppred

# Example setup to test the function
k = 0
zmeas = np.array([0.0101])
u = np.array([0.0101, 0.1188])
P = np.array([[0.00516961, 0.008445032], [0.008445032, 2.02692169]])
V = np.array([0, 0])
zcov = np.array([0.01])
H = np.array([[1, 0]])
F = np.array([[1.0191, 0.0099], [-0.2474, 0.9994]])
Qk = np.array([[0.002, 0.002], [0.002, 0.438]])

# Instantiate trackingKF filter
filter = trackingKF(F, H, u, P, Qk)

# Test the id_predict function
xpred, Ppred = id_predict(filter)

print("Predicted State (xpred):", xpred)
print("Predicted Covariance (Ppred):", Ppred)
