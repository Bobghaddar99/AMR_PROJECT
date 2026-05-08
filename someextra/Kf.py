import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1. Create True Trajectory + Noisy Measurements
# -----------------------------------------------------------

np.random.seed(0)        # For reproducibility
T = 40                   # Number of timesteps(t)

# True robot motion (smooth straight line, as we have divided the line into multiple timestamps)
true_pos = np.linspace(2, 12, T)

# Noisy measurements (Gaussian noise: mean=0, std=0.5)
measurements = true_pos + np.random.normal(0, 100, T)

print("First 5 true positions:", true_pos[:5])
print("First 5 noisy measurements:", measurements[:5])

# -----------------------------------------------------------
# 2. Initialize Kalman Filter variables
# -----------------------------------------------------------

x_est = np.zeros(T)      # Filtered estimates (posterior)
x_pred_all = np.zeros(T) # Predictions before seeing measurement (prior)
P = np.zeros(T)          # Uncertainty (variance)

# Initial conditions
x_est[0] = 1.0            # Initial guess of position
P[0] = 2.0              # Initial uncertainty

# Noise parameters
Q = 0.02                 # Process noise variance
R = 10                # Measurement noise variance

print("\nInitial guess x0 =", x_est[0])
print("Initial uncertainty P0 =", P[0])

# -----------------------------------------------------------
# 3. Kalman Filter Loop (Prediction + Update)
# -----------------------------------------------------------

for t in range(1, T):

    # -------------------------
    # (A) Prediction Step
    # -------------------------

    # Motion model: x_t = x_{t-1}  (simple constant position model)
    x_pred = x_est[t-1]

    # Predicted uncertainty: P + Q
    P_pred = P[t-1] + Q

    # Store for plotting
    x_pred_all[t] = x_pred

    # -------------------------
    # (B) Update Step
    # -------------------------

    # Kalman Gain
    K = P_pred / (P_pred + R)

    # Update estimate with measurement
    x_est[t] = x_pred + K * (measurements[t] - x_pred)

    # Update uncertainty
    P[t] = (1 - K) * P_pred

    # Print the first few steps for understanding
    if t < 5:
        print(f"\nStep {t}:")
        print(f"  Prediction      = {x_pred:.2f}")
        print(f"  Measurement     = {measurements[t]:.2f}")
        print(f"  KF Update       = {x_est[t]:.2f}")
        print(f"  Kalman Gain K   = {K:.2f}")
        print(f"  Uncertainty P   = {P[t]:.3f}")

# -----------------------------------------------------------
# 4. Plot Results
# -----------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(true_pos, label="True Position")
plt.plot(measurements, "o", alpha=0.5, label="Measurements")
plt.plot(x_pred_all, "--", label="Predictions (before update)")
plt.plot(x_est, label="KF Estimate (after update)")

plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.title("Kalman Filter Result (1D Position Tracking)")
plt.grid(True)
plt.show()
