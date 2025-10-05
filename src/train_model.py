# src/train_model.py
# import numpy as np
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.metrics import accuracy_score
# import joblib
# import os

# # ----------------------------
# # 1. Load the image-sensing data
# # ----------------------------
# digits = load_digits()
# X, y = digits.data, digits.target

# # Normalize pixel values (0–16 → 0–1)
# X = X / 16.0

# # ----------------------------
# # 2. Split into training & test sets
# # ----------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # ----------------------------
# # 3. Define ANN model
# # ----------------------------
# model = MLPClassifier(
#     hidden_layer_sizes=(64, 32),
#     activation='relu',
#     solver='adam',
#     max_iter=300,
#     random_state=42
# )

# # ----------------------------
# # 4. Train the network
# # ----------------------------
# print("Training ANN on image-sensing data...")
# model.fit(X_train, y_train)

# # ----------------------------
# # 5. Evaluate
# # ----------------------------
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"✅ Accuracy: {acc * 100:.2f}%")

# # ----------------------------
# # 6. Save the model
# # ----------------------------
# os.makedirs("../model", exist_ok=True)
# joblib.dump(model, "../model/ann_model.pkl")
# print("Model saved successfully to /model/ann_model.pkl")

# src/train_model.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# --- Model Path Update ---
MODEL_PATH = "../model/ann_model_99_percent_acc.pkl"

# ----------------------------
# 1. Load the image-sensing data
# ----------------------------
digits = load_digits()
X, y = digits.data, digits.target

# Normalize pixel values (0–16 → 0–1)
X = X / 16.0

# ----------------------------
# 2. Split into training & test sets
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 3. Define ANN model (High-Accuracy Configuration)
# ----------------------------
model = MLPClassifier(
    # IMPROVEMENT 1: Increased layers (Wider and deeper architecture)
    hidden_layer_sizes=(100, 100, 50),
    activation='relu',
    # IMPROVEMENT 2: Change solver to 'lbfgs' for small, clean data (BEST for accuracy)
    solver='lbfgs',
    # IMPROVEMENT 3: Increased max_iter for full convergence
    max_iter=800,
    random_state=42
)

# ----------------------------
# 4. Train the network
# ----------------------------
print("Training ultra-high accuracy ANN on image-sensing data...")
# Note: 'lbfgs' trains silently unless you pass verbose=True (which can be messy)
model.fit(X_train, y_train)

# ----------------------------
# 5. Evaluate
# ----------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Final Accuracy: {acc * 100:.2f}% (Targeting 99%+) ")

# ----------------------------
# 6. Save the model
# ----------------------------
os.makedirs("../model", exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved successfully to {MODEL_PATH}")





# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import os

# # ----------------------------
# # 1. Load and normalize data
# # ----------------------------
# digits = load_digits()
# X, y = digits.data, digits.target
# X = X / 16.0  # normalize to 0-1

# # ----------------------------
# # 2. Split data
# # ----------------------------
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features (helps MLP and SVM)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # ----------------------------
# # 3. Define models
# # ----------------------------
# models = {
#     "Logistic Regression": LogisticRegression(max_iter=500),
#     "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
#     "Decision Tree": DecisionTreeClassifier(),
#     "SVM": SVC(),
#     "MLP Classifier": MLPClassifier(hidden_layer_sizes=(64, 32),
#                                     activation='relu',
#                                     solver='adam',
#                                     max_iter=300,
#                                     random_state=42)
# }

# # ----------------------------
# # 4. Train & evaluate models
# # ----------------------------
# accuracies = {}

# # Optional: dynamic traditional baseline
# # Here we assume traditional method has 75% accuracy
# # You could replace this with a function that computes it dynamically
# accuracies["Traditional Baseline"] = 0.75

# for name, model in models.items():
#     print(f"Training {name}...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     accuracies[name] = acc
#     print(f"{name} Accuracy: {acc*100:.2f}%\n")

# # ----------------------------
# # 5. Plot results professionally
# # ----------------------------
# plt.style.use('ggplot')
# plt.figure(figsize=(11,6))
# names = list(accuracies.keys())
# scores = list(accuracies.values())
# colors = ['#FFA500','#FF6F61','#6B5B95','#88B04B','#F7CAC9','#92A8D1']  # include color for baseline

# bars = plt.bar(names, scores, color=colors, edgecolor='black')
# plt.ylim(0,1)
# plt.ylabel("Accuracy")
# plt.title("Different Models Accuracy rate\n",
#           fontsize=12, fontweight='bold')

# # Display accuracy above each bar
# for bar, score in zip(bars, scores):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{score*100:.2f}%", ha='center', fontweight='bold')

# plt.xticks(rotation=30)
# plt.tight_layout()

# # ----------------------------
# # 6. Save chart as image
# # ----------------------------
# os.makedirs("charts", exist_ok=True)
# plt.savefig("charts/model_accuracy_comparison_with_baseline.png", dpi=300)
# print("✅ Chart saved as 'charts/model_accuracy_comparison_with_baseline.png'")

# plt.show()

