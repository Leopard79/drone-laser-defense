"""
01_intuition_1d.py
هدف الملف: فهم مبدأ كالمان فلتر بأبسط صورة ممكنة — بعد واحد (1D)، بدون أي مكتبة.
السيناريو: نقطة بتتحرك بسرعة شبه ثابتة، وعندنا قياسات مهتزة (فيها ضجيج) لموقعها.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. توليد بيانات مصطنعة (الحقيقة + القياس المهتز) ----------

np.random.seed(42)  # عشان النتيجة تكون نفسها كل مرة تشغل الكود (قابلة لإعادة الإنتاج)

n_steps = 50          # عدد الفريمات/الخطوات الزمنية
dt = 1.0               # الزمن بين كل خطوة وخطوة (بالثانية) — ثابت هون للتبسيط
true_velocity = 2.0    # السرعة الحقيقية (م/ثا) — نحن عارفينها لأنها بيانات مصطنعة، بالواقع ما منعرفها

true_positions = []     # المسار الحقيقي (لو ما في ضجيج إطلاقاً)
measured_positions = [] # المسار المقاس (فيه ضجيج، متل ما بيطلع من YOLO بالواقع)

true_x = 0.0
for i in range(n_steps):
    true_x += true_velocity * dt
    true_positions.append(true_x)

    noise = np.random.normal(0, 1.5)   # ضجيج عشوائي بانحراف معياري 1.5 — يمثل عدم دقة القياس
    measured_positions.append(true_x + noise)

true_positions = np.array(true_positions)
measured_positions = np.array(measured_positions)

# ---------- 2. تطبيق كالمان فلتر اليدوي (1D) ----------

# -- الحالة الابتدائية: ما نعرف شي أكيد، فنبلش بتخمين بسيط --
x_est = 0.0      # تقدير الموقع (الحالة اللي عم نتابعها)
v_est = 0.0      # تقدير السرعة (كمان جزء من الحالة، منقدرها بنفس الوقت)

# -- عدم اليقين الابتدائي (كم نحن متأكدين من x_est, v_est) --
P = 100.0        # قيمة كبيرة = "ما نحن متأكدين إطلاقاً بالبداية"، وهاد طبيعي بأول فريم

# -- معاملات الضجيج (هاي اللي رح "توليفها" لاحقاً متل توليف PID) --
Q = 0.05         # Process Noise: مدى ثقتنا بالنموذج الحركي (قيمة صغيرة = نثق كتير بأن الحركة شبه منتظمة)
R = 2.0          # Measurement Noise: مدى ثقتنا بدقة القياس (قيمة كبيرة = القياس مهتز/غير موثوق)

estimated_positions = []
estimated_velocities = []

for z in measured_positions:   # z = القياس الجديد اللي وصل هالخطوة (رمز شائع بأدبيات كالمان للقياس)

    # ===== خطوة Predict =====
    # توقع الموقع الجديد بناءً على آخر موقع وآخر سرعة معروفة
    x_pred = x_est + v_est * dt
    v_pred = v_est  # نفترض السرعة ما بتتغير فجأة (Constant Velocity Model)

    # عدم اليقين بيزيد كل خطوة توقع (لأننا ما عندنا معلومة جديدة بعد، بس عم نتوقع)
    P_pred = P + Q

    # ===== خطوة Update =====
    # Kalman Gain: كم نثق بالقياس الجديد مقارنة بتوقعنا
    # لو R كبيرة (قياس سيء) K بيصغر، ولو P_pred كبيرة (توقعنا غير أكيد) K بيكبر
    K = P_pred / (P_pred + R)

    # التصحيح: نمزج التوقع مع القياس الفعلي حسب الوزن K
    innovation = z - x_pred          # الفرق بين القياس والتوقع — اسمه "Innovation" بأدبيات كالمان
    x_est = x_pred + K * innovation
    v_est = v_pred + (K * innovation) / dt   # تحديث تقدير السرعة كمان (تبسيط مقصود هون)

    # تحديث عدم اليقين — بعد ما استفدنا من قياس جديد، عدم اليقين بيقل
    P = (1 - K) * P_pred

    estimated_positions.append(x_est)
    estimated_velocities.append(v_est)

estimated_positions = np.array(estimated_positions)

# ---------- 3. الرسم البياني — قارن بعينك ----------

plt.figure(figsize=(10, 6))
plt.plot(true_positions, label="real position", linewidth=2)
plt.plot(measured_positions, label="measured position", linestyle="--", alpha=0.6)
plt.plot(estimated_positions, label="estimated position (filtered)", linewidth=2)
plt.xlabel("الخطوة الزمنية (frame)")
plt.ylabel("الموقع (متر)")
plt.title("كالمان فلتر 1D: القياس الخام مقابل التقدير المصفى")
plt.legend()
plt.grid(True)
plt.savefig("01_result.png", dpi=120)
plt.show()

print("آخر سرعة مقدّرة:", round(v_est, 3), "م/ثا (الحقيقية كانت:", true_velocity, ")")