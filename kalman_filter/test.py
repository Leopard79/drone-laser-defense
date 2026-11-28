from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

steps = 50
ideal_velocity = 2.0
dt = 1

ideal_pos = []
measured_pos = []

ideal_x = 0.0
for i in range(steps):
    ideal_x += ideal_velocity * dt
    ideal_pos.append(ideal_x)

    noise = np.random.normal(0,1.5)
    measured_pos.append(ideal_x + noise)

ideal_pos = np.array(ideal_pos)
measured_pos = np.array(measured_pos)

x_est = 0.0
v_est = 0.0

P = 100.0
Q = 0.05
R = 2.0

est_pos = []
est_vel = []

for z in measured_pos:
    x_pred = x_est + v_est * dt
    v_pred = v_est 

    P_pred = P + Q
    K = P_pred / (P_pred + R)

    x_est = x_pred + K * (z - x_pred)
    v_est = v_pred + K * (z - x_pred) / dt  

    P = (1 - K) * P_pred

    est_pos.append(x_est)
    est_vel.append(v_est)   

est_pos = np.array(est_pos)

plt.figure(figsize=(10,6))
plt.plot(ideal_pos, label='Ideal Position', color='g')  
plt.plot(measured_pos, label='Measured Position', color='r')
plt.plot(est_pos, label='Estimated Position', color='b')
plt.legend()
plt.show()