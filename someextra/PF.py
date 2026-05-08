import numpy as np
import math

m = 3  # number of particles
L = [2.0, 2.0]  # landmark
R = 0.5  # measurement noise variance

error = np.zeros(m)
unnorm = np.zeros(m)
norm = np.zeros(m)
measurements = np.zeros(m)

Particles = [(1.0, 2.0), (2.0, 3.0), (0.0, 2.0)]
zt = 1.0  # measurement

# 1. 
def measurement_model(x):
    return np.sqrt((x[0] - L[0])**2 + (x[1] - L[1])**2)

def error_model(z_pred):
    return zt - z_pred

# Compute predicted measurements + errors
for i in range(m):
    measurements[i] = measurement_model(Particles[i])
    error[i] = error_model(measurements[i])

# 2+3 Compute unnormalized weights
for i in range(m):
    err = error[i]
    expo = -(err**2) / (2 * R)    # Gaussian denominator should be 2*R
    unnorm[i] = math.exp(expo)

# Normalize weights
total = np.sum(unnorm)
norm = unnorm / total

# 4 Compute weighted mean state
x = 0
y = 0

for i in range(m):
    x += norm[i] * Particles[i][0]
    y += norm[i] * Particles[i][1]

