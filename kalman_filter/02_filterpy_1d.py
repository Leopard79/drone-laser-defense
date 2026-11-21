"""
02_filterpy_1d.py
نفس سيناريو الملف الأول بالضبط (نقطة بسرعة شبه ثابتة + قياسات مهتزة)،
بس هلق باستخدام مكتبة filterpy بدل الحسابات اليدوية.
الهدف: تتعلم شو هي الحقول اللي لازم تعبيها بالمكتبة، وتتأكد إنها بتطلع نفس نتيجة الكود اليدوي تقريباً.
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# ---------- 1. نفس البيانات المصطنعة بالضبط من الملف الأول ----------

np.random.seed(42)

n_steps = 50
dt = 1.0
true_velocity = 2.0

true_positions = []
measured_positions = []

true_x = 0.0
for i in range(n_steps):
    true_x += true_velocity * dt
    true_positions.append(true_x)
    noise = np.random.normal(0, 1.5)
    measured_positions.append(true_x + noise)

true_positions = np.array(true_positions)
measured_positions = np.array(measured_positions)

# ---------- 2. إعداد كالمان فلتر بـ filterpy ----------

# dim_x = 2 → الحالة عندنا بعدين: [position, velocity]
# dim_z = 1 → القياس بعد واحد بس: [position]
kf = KalmanFilter(dim_x=2, dim_z=1)

# -- الحالة الابتدائية x: نفس x_est, v_est من قبل --
kf.x = np.array([0.0, 0.0])   # [position, velocity] ابتدائي

# -- F: State Transition Matrix (نموذج الحركة) --
kf.F = np.array([[1.0, dt],
                  [0.0, 1.0]])

# -- H: Measurement Matrix (كيف نحوّل الحالة لشكل القياس) --
kf.H = np.array([[1.0, 0.0]])

# -- P: عدم اليقين الابتدائي --
kf.P *= 100.0

# -- Q: Process Noise --
kf.Q = np.array([[0.05, 0.0],
                  [0.0, 0.05]])

# -- R: Measurement Noise --
kf.R = np.array([[2.0]])

# ---------- 3. حلقة Predict + Update ----------

estimated_positions = []
estimated_velocities = []

for z in measured_positions:
    kf.predict()           # يحسب x_pred داخلياً (خطوة Predict)
    kf.update(z)           # يمزج القياس الجديد z مع x_pred (خطوة Update)

    estimated_positions.append(kf.x[0])
    estimated_velocities.append(kf.x[1])

estimated_positions = np.array(estimated_positions)

# ---------- 4. الرسم ----------

plt.figure(figsize=(10, 6))
plt.plot(true_positions, label="الموقع الحقيقي", linewidth=2)
plt.plot(measured_positions, label="القياس الخام (مهتز)", linestyle="--", alpha=0.6)
plt.plot(estimated_positions, label="تقدير filterpy (مصفى)", linewidth=2)
plt.xlabel("الخطوة الزمنية (frame)")
plt.ylabel("الموقع (متر)")
plt.title("كالمان فلتر 1D بـ filterpy")
plt.legend()
plt.grid(True)
plt.savefig("02_result.png", dpi=120)
plt.show()

print("آخر سرعة مقدّرة:", round(estimated_velocities[-1], 3), "م/ثا")