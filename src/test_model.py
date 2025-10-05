# src/test_model.py
# import joblib
# from sklearn.datasets import load_digits
# import matplotlib.pyplot as plt
# import random

# # Load data & model
# digits = load_digits()
# model = joblib.load("../model/ann_model.pkl")

# # Pick random image
# i = random.randint(0, len(digits.images) - 1)
# image = digits.images[i]
# label = digits.target[i]

# # Flatten and predict
# pred = model.predict([image.flatten() / 16.0])[0]

# plt.imshow(image, cmap="gray")
# plt.title(f"True: {label} | Predicted: {pred}")
# plt.show()
# src/test_model.py
import joblib
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import random

# Load data & the NEW high-accuracy model
digits = load_digits()
# --- CHANGE HERE ---
model = joblib.load("../model/ann_model_99_percent_acc.pkl") # Use the new model file

# Pick random image
i = random.randint(0, len(digits.images) - 1)
image = digits.images[i]
label = digits.target[i]

# Flatten and predict
# The scikit-learn digits dataset has values from 0-16
pred = model.predict([image.flatten() / 16.0])[0]

plt.imshow(image, cmap="gray")
plt.title(f"True: {label} | Predicted: {pred}")
plt.show()