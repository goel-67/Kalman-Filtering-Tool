import numpy as np
from Mupdate import mupdate
from INFtoCOV import inf_to_cov

# Define the trackingKF class as previously provided
class trackingKF:
    def __init__(self, F, H, state, state_covariance, process_noise):
        self.StateTransitionModel = F
        self.MeasurementModel = H
        self.State = state
        self.StateCovariance = state_covariance
        self.ProcessNoise = process_noise

# Define the IDcorrect function as previously provided
def id_correct(filter, zmeas, zcov):
    u = filter.State
    P = filter.StateCovariance
    H = filter.MeasurementModel
    F = filter.StateTransitionModel
    Qk = filter.ProcessNoise
    M = Qk.shape
    Vzeros = np.zeros((M[0], 1))
    I = np.eye(M[0])

    # Perform measurement update
    u, V, B = mupdate(0, zmeas, u, P, Vzeros, zcov, H)
    print("Corrected state (u):", u)
    print("V:", V)
    print("B:", B)

    # Get the shape of B and set up L
    col_L, row_L = B.shape
    L = col_L
    B = inf_to_cov(V, B, L)

    # Set corrected values
    xcorr = u
    Pcorr = B

    return xcorr, Pcorr

# Test parameters from Table 8
zmeas = np.array([0.0101])
u = np.array([[0.0101], [0.1188]])
P = np.array([[0.01071225, 0.017495523], [0.017495523, 2.04175521]])
V = np.array([[0], [0]])
zcov = np.array([[0.01]])
H = np.array([[1, 0]])
F = np.array([[1.0191, 0.0099], [-0.2474, 0.9994]])
Qk = np.array([[0.002, 0.002], [0.002, 0.438]])

# Initialize filter object
filter = trackingKF(F, H, u, P, Qk)

# Run IDcorrect function
xcorr, Pcorr = id_correct(filter, zmeas, zcov)

print("Corrected state (xcorr):", xcorr)
print("Corrected covariance (Pcorr):", Pcorr)
