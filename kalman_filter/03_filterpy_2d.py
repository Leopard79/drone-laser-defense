"""
03_filterpy_2d.py
توسيع الفكرة لبعدين (x, y) بدل بعد واحد — زي موقع الدرون الحقيقي بالصورة.
الحالة صارت 4 عناصر: [x, y, vx, vy]
القياس بقى عنصرين بس: [x, y]  (متل ما YOLO بيديك مركز bounding box، مش سرعته)
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter

# ---------- 1. توليد مسار مصطنع ثنائي الأبعاد ----------

np.random.seed(42)

n_steps = 50
dt = 1.0

# نفترض جسم ماشي بخط مستقيم مائل: سرعة ثابتة بكل من x و y
vx_true = 2.0
vy_true = 1.0

true_x, true_y = 0.0, 0.0
true_positions = []      # هون كل عنصر رح يكون زوج (x, y)
measured_positions = []

for i in range(n_steps):
    true_x += vx_true * dt
    true_y += vy_true * dt
    true_positions.append([true_x, true_y])

    # ضجيج منفصل لكل محور (x وy كل واحد فيهم بيهتز لحاله)
    noise_x = np.random.normal(0, 1.5)
    noise_y = np.random.normal(0, 1.0)   # خليت ضجيج y أقل، كمثال على اختلاف دقة المحورين
    measured_positions.append([true_x + noise_x, true_y + noise_y])

true_positions = np.array(true_positions)
measured_positions = np.array(measured_positions)

# ---------- 2. إعداد كالمان فلتر لحالة 4 أبعاد ----------

# dim_x=4 → الحالة: [x, y, vx, vy]
# dim_z=2 → القياس: [x, y] بس
kf = KalmanFilter(dim_x=4, dim_z=2)

kf.x = np.array([0.0, 0.0, 0.0, 0.0])   # [x, y, vx, vy] ابتدائي

# F: نموذج الحركة — كل محور (x,y) بيتحدث بنفس منطق البعد الواحد، بس الاثنين سوا
kf.F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1,  0],
                  [0, 0, 0,  1]])

# H: من بين (x, y, vx, vy)، خد بس (x, y) للمقارنة مع القياس
kf.H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

kf.P *= 100.0   # عدم يقين ابتدائي كبير لكل الحالة الأربعة

# Q: ثقتنا بنموذج الحركة لكل متغير (x, y, vx, vy) — صغيرة لأننا واثقين بافتراض السرعة الثابتة
kf.Q = np.eye(4) * 0.05   # np.eye(4) = مصفوفة وحدة 4x4، أسهل من كتابتها يدوياً

# R: ثقتنا بدقة القياس على كل محور — لاحظ اخترنا قيم مختلفة (x أقل دقة من y)
kf.R = np.array([[2.0, 0.0],
                  [0.0, 1.0]])

# ---------- 3. حلقة Predict + Update ----------

estimated_positions = []

for z in measured_positions:
    kf.predict()
    kf.update(z)            # z هلق زوج [x, y]، مش رقم واحد متل قبل
    estimated_positions.append([kf.x[0], kf.x[1]])   # ناخد بس x, y (نتجاهل vx, vy بالرسم)

estimated_positions = np.array(estimated_positions)

# ---------- 4. الرسم — هون منرسم المسار بالفضاء (x,y) مش بالزمن ----------

plt.figure(figsize=(8, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], label="المسار الحقيقي", color="g", linewidth=2)
plt.scatter(measured_positions[:, 0], measured_positions[:, 1], label="القياسات (مهتزة)", color="b", alpha=0.4, s=20)
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], label="التقدير المصفى", color="r", linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.title("كالمان فلتر 2D — المسار بفضاء (x, y)")
plt.legend()
plt.grid(True)
plt.axis("equal")   # عشان المحورين ياخدوا نفس المقياس، ما يصير تشويه بصري بالشكل
plt.show()

print("آخر سرعة مقدّرة vx:", round(kf.x[2], 3), "(الحقيقية:", vx_true, ")")
print("آخر سرعة مقدّرة vy:", round(kf.x[3], 3), "(الحقيقية:", vy_true, ")")