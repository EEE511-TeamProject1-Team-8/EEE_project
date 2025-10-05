import os
import ssl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

# ----------------------------
# Model path
# ----------------------------
MODEL_PATH = "model/ann_model_28x28_ultra_acc.pkl"

# ----------------------------
# Train ANN if model not found
# ----------------------------
if not os.path.exists(MODEL_PATH):
    print("Training ultra-high-accuracy ANN model on MNIST...")

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data, mnist.target.astype(int)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ann = MLPClassifier(hidden_layer_sizes=(512, 256, 128), activation='relu',
                        solver='adam', max_iter=600, verbose=True)
    ann.fit(X_train, y_train)

    os.makedirs(os.path.dirname(MODEL_PATH) or '.', exist_ok=True)
    joblib.dump(ann, MODEL_PATH)
    print(f"✅ Model saved at {MODEL_PATH}")

# ----------------------------
# Load model
# ----------------------------
model = joblib.load(MODEL_PATH)

# ----------------------------
# Create drawing canvas
# ----------------------------
canvas_size = 28
canvas = np.zeros((canvas_size, canvas_size))

fig, (ax, ax_preview) = plt.subplots(1, 2, figsize=(6, 3))
plt.subplots_adjust(bottom=0.2)

im = ax.imshow(canvas, cmap='gray', vmin=0, vmax=1)
ax.axis('off')
ax.set_title("Draw a digit")

preview_data = np.zeros((28, 28))
im_preview = ax_preview.imshow(preview_data, cmap='gray', vmin=0, vmax=1)
ax_preview.axis('off')
ax_preview.set_title("ANN input preview")

drawing = False

# ----------------------------
# Drawing helper
# ----------------------------
def draw_at(x, y, size=2):
    """Draw a circular brush stroke."""
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            xi, yj = int(x) + i, int(y) + j
            if 0 <= xi < canvas_size and 0 <= yj < canvas_size:
                dist_sq = i ** 2 + j ** 2
                if dist_sq <= size ** 2:
                    intensity = 1.0 - (dist_sq / (size ** 2 + 1)) * 0.5
                    canvas[yj, xi] = np.clip(canvas[yj, xi] + intensity * 0.5, 0, 1)

# ----------------------------
# Prediction function
# ----------------------------
def predict_digit():
    """Preprocess and predict."""
    smoothed = gaussian_filter(canvas, sigma=0.8)
    resized = resize(smoothed, (28, 28), anti_aliasing=True)

    # Centering logic
    rows = np.any(resized > 0.1, axis=1)
    cols = np.any(resized > 0.1, axis=0)
    if rows.any() and cols.any():
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        digit = resized[ymin:ymax+1, xmin:xmax+1]
        h, w = digit.shape
        scale = 20.0 / max(h, w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        resized_digit = resize(digit, (new_h, new_w), anti_aliasing=True)
        centered = np.zeros((28, 28))
        top = (28 - new_h) // 2
        left = (28 - new_w) // 2
        centered[top:top+new_h, left:left+new_w] = resized_digit
    else:
        centered = resized

    # Normalize and (optional) invert if needed
    if np.mean(centered) > 0.5:  # too bright means white background
        centered = 1 - centered

    centered = centered / np.max(centered) if np.max(centered) > 0 else centered

    flat = centered.flatten().reshape(1, -1)
    pred = model.predict(flat)[0]
    return pred, centered

# ----------------------------
# Mouse event callbacks
# ----------------------------
def on_press(event):
    global drawing
    drawing = True

def on_release(event):
    global drawing
    drawing = False

def on_motion(event):
    if drawing and event.xdata is not None and event.ydata is not None:
        draw_at(event.xdata, event.ydata)
        im.set_data(canvas)
        pred, preview = predict_digit()
        im_preview.set_data(preview)
        ax.set_title(f"Live Prediction: {pred}")
        fig.canvas.draw_idle()

# ----------------------------
# Clear canvas button
# ----------------------------
def clear_canvas(event):
    global canvas
    canvas = np.zeros((canvas_size, canvas_size))
    im.set_data(canvas)
    im_preview.set_data(np.zeros((28, 28)))
    ax.set_title("Draw a digit")
    fig.canvas.draw_idle()

ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
button = Button(ax_button, 'Clear Canvas')
button.on_clicked(clear_canvas)

# ----------------------------
# Connect mouse events
# ----------------------------
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.show()

