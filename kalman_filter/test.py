from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

steps = 50
dt = 1.0
ideal_velocity = 2.0

ideal_positions = []
measured_positions = []

ideal_x = 0.0

for i in range(steps):
    ideal_x += ideal_velocity * dt
    ideal_positions.append(ideal_x)

    noise = np.random.normal(0,1.5)
    measured_positions.append(ideal_x + noise)

measured_positions = np.array(measured_positions)
ideal_positions = np.array(ideal_positions)

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([0.0, 0.0])

kf.F = np.array([[1.0, dt],
                 [0.0, 1.0]])

kf.H = np.array([[1.0, 0.0]])

kf.P *= 100.0

kf.Q = np.array([[0.05,0.0],[0.0,0.05]])
kf.R = np.array([[2.0]])
estimated_positions = []
estimated_velocities = []
for z in measured_positions:
    kf.predict()
    kf.update(z)
    estimated_positions.append(kf.x[0])
    estimated_velocities.append(kf.x[1])

estimated_positions = np.array(estimated_positions)
estimated_velocities = np.array(estimated_velocities)

plt.figure(figsize=(10,6))
plt.plot(ideal_positions,label='ideal position',color='g')
plt.plot(measured_positions,label='measured position',color='b')
plt.plot(estimated_positions,label='estimated position',color='r')
plt.legend()
plt.show()
