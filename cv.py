# =============================================================================
# LAB 1 — Basic Image Processing Operations
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Read & Display ---
image = cv2.imread("image.png")           # replace with your image path
h, w = image.shape[:2]
small = cv2.resize(image, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
plt.imshow(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()

# --- Format Conversion & File Size Comparison ---
cv2.imwrite("image_jpg.jpg", image)
cv2.imwrite("image_bmp.bmp", image)
cv2.imwrite("image_jpeg.jpeg", image)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, path, label in zip(axes,
    ["image.png", "image_jpg.jpg", "image_bmp.bmp", "image_jpeg.jpeg"],
    ["PNG", "JPG", "BMP", "JPEG"]):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    size = os.path.getsize(path) / 1024
    ax.imshow(img); ax.set_title(f"{label}\n{size:.2f} KB"); ax.axis("off")
plt.tight_layout(); plt.show()

# --- Grayscale Conversion ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.resize(gray, (w // 2, h // 2)), cmap="gray"); plt.axis("off"); plt.show()

# --- Enhancement Operations ---
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_float = img_rgb / 255.0

mean_blur     = cv2.blur(img_rgb, (5, 5))
gaussian_blur = cv2.GaussianBlur(img_rgb, (5, 5), 0)
median_blur   = cv2.medianBlur(img_rgb, 5)
equalized     = cv2.equalizeHist(gray)

c = 1 / np.log(1 + np.max(img_float))
log_t   = np.uint8(c * np.log(1 + img_float) * 255)
gamma_t = np.uint8(np.power(img_float, 0.5) * 255)

titles = ["Log Transform", "Gamma (0.5)", "Hist EQ",
          "Mean Blur", "Gaussian Blur", "Median Blur"]
imgs   = [log_t, gamma_t, equalized, mean_blur, gaussian_blur, median_blur]
cmaps  = [None, None, "gray", None, None, None]

plt.figure(figsize=(16, 8))
for i, (im, t, cm) in enumerate(zip(imgs, titles, cmaps)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(im, cmap=cm); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()

# --- Geometric Transformations ---
center = (w // 2, h // 2)

M_rot   = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(image, M_rot, (w, h))

scaled = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

M_trans    = np.float32([[1, 0, 100], [0, 1, 50]])
translated = cv2.warpAffine(image, M_trans, (w, h))

sf = 0.3
M_shear = np.float32([[1, sf, 0], [0, 1, 0]])
sheared = cv2.warpAffine(image, M_shear, (int(w + h * sf), h))

flipped = cv2.flip(image, 1)

geo_imgs   = [image, rotated, scaled, translated, sheared, flipped]
geo_titles = ["Original", "Rotated 45°", "Scaled 1.5x",
              "Translated (100,50)", "Sheared 0.3", "Flipped H"]

plt.figure(figsize=(18, 8))
for i, (im, t) in enumerate(zip(geo_imgs, geo_titles)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()


# =============================================================================
# LAB 2 — Image Enhancement using Spatial Domain Filtering
# =============================================================================

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread("image.jpg")              # replace with your image path
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB)); plt.axis("off"); plt.show()

# --- Gaussian Noise + Mean Blur ---
def add_gaussian_noise(image, mean=0, std_dev=25):
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.uint8)
    return cv.add(image, noise)

noisy = add_gaussian_noise(img, std_dev=1)
averaged = cv.blur(noisy, (5, 5))

plt.figure(figsize=(15, 5))
for i, (im, t) in enumerate(zip([img, noisy, averaged],
                                  ["Original", "Noisy", "Mean Blurred (5x5)"])):
    plt.subplot(1, 3, i + 1); plt.imshow(im); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()

# --- Gaussian Blur (effect of sigma) ---
gb0 = cv.GaussianBlur(noisy, (5, 5), 0)
gb1 = cv.GaussianBlur(noisy, (5, 5), 1)

plt.figure(figsize=(15, 5))
for i, (im, t) in enumerate(zip([img, noisy, gb0, gb1],
                                  ["Original", "Noisy", "Gaussian σ=0", "Gaussian σ=1"])):
    plt.subplot(1, 4, i + 1); plt.imshow(im); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()

# --- Salt & Pepper Noise + Median Filter ---
def add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy = image.copy()
    total = image.size
    coords = [np.random.randint(0, i, int(total * salt_prob)) for i in image.shape]
    noisy[coords[0], coords[1]] = 255
    coords = [np.random.randint(0, i, int(total * pepper_prob)) for i in image.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy

sp_noisy = add_salt_pepper_noise(img)
med3 = cv.medianBlur(sp_noisy, 3)
med5 = cv.medianBlur(sp_noisy, 5)

plt.figure(figsize=(15, 5))
for i, (im, t) in enumerate(zip([img, sp_noisy, med3, med5],
                                  ["Original", "S&P Noise", "Median 3x3", "Median 5x5"])):
    plt.subplot(1, 4, i + 1); plt.imshow(im); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()

# --- Edge Detection: Sobel (1st derivative) ---
gray_img = cv.cvtColor(med5, cv.COLOR_BGR2GRAY)
sobelx   = cv.Sobel(gray_img, cv.CV_64F, 1, 0, ksize=5)
sobely   = cv.Sobel(gray_img, cv.CV_64F, 0, 1, ksize=5)
sobel_combined = cv.magnitude(sobelx, sobely)

plt.figure(figsize=(15, 5))
for i, (im, t, cm) in enumerate(zip(
        [gray_img, sobelx, sobely, sobel_combined],
        ["Grayscale", "Sobel X", "Sobel Y", "Sobel Combined"],
        ["gray", "gray", "gray", "gray"])):
    plt.subplot(1, 4, i + 1); plt.imshow(im, cmap=cm); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()

# --- Edge Detection: Laplacian (2nd derivative) ---
laplacian = cv.Laplacian(gray_img, cv.CV_64F)
positive  = cv.bitwise_not(laplacian)

plt.figure(figsize=(12, 4))
for i, (im, t) in enumerate(zip([gray_img, laplacian, positive],
                                  ["Grayscale", "Laplacian", "Normalized"])):
    plt.subplot(1, 3, i + 1); plt.imshow(im, cmap="gray"); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()

# --- Laplacian of Gaussian (LoG) ---
blurred_log = cv.GaussianBlur(gray_img, (5, 5), 0)
log_img     = cv.Laplacian(blurred_log, cv.CV_64F)
neg_log     = cv.bitwise_not(log_img)

plt.figure(figsize=(16, 4))
for i, (im, t) in enumerate(zip([gray_img, blurred_log, log_img, neg_log],
                                  ["Grayscale", "Gaussian Blurred", "LoG", "Negative LoG"])):
    plt.subplot(1, 4, i + 1); plt.imshow(im, cmap="gray"); plt.title(t); plt.xticks([]); plt.yticks([])
plt.show()


# =============================================================================
# LAB 3 — Image Segmentation: Thresholding & K-Means
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("image.tif")           # replace with your image path
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.show()

# --- Otsu's Thresholding ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu threshold: {ret}")

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(gray, cmap="gray"); plt.title("Grayscale"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(thresh_otsu, cmap="gray")
plt.title(f"Otsu Threshold = {ret:.2f}"); plt.axis("off")
plt.subplot(1, 3, 3)
plt.hist(gray.ravel(), 256, [0, 256], color="gray")
plt.axvline(x=ret, color="r", linestyle="--", label=f"Otsu ({ret:.0f})")
plt.title("Histogram"); plt.xlabel("Intensity"); plt.ylabel("Count"); plt.legend()
plt.tight_layout(); plt.show()

# --- Effect of Different Thresholds ---
threshold_values = [100, 150, ret, 200]
plt.figure(figsize=(16, 4))
plt.subplot(1, 5, 1); plt.imshow(gray, cmap="gray"); plt.title("Original"); plt.axis("off")
for i, T in enumerate(threshold_values):
    _, t_img = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)
    plt.subplot(1, 5, i + 2); plt.imshow(t_img, cmap="gray")
    plt.title(f"T = {T:.0f}"); plt.axis("off")
plt.suptitle("Effect of Threshold Value", fontsize=14)
plt.tight_layout(); plt.show()

# --- K-Means Segmentation ---
image_km = cv2.imread("image_color.png")  # replace with your color image path
image_km_rgb = cv2.cvtColor(image_km, cv2.COLOR_BGR2RGB)

pixel_vals = np.float32(image_km_rgb.reshape((-1, 3)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k_values = [3, 5, 10]

plt.figure(figsize=(18, 5))
plt.subplot(1, 4, 1); plt.imshow(image_km_rgb); plt.title("Original"); plt.axis("off")

for i, k in enumerate(k_values):
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    seg = centers[labels.flatten()].reshape(image_km_rgb.shape)
    plt.subplot(1, 4, i + 2); plt.imshow(seg); plt.title(f"K = {k}"); plt.axis("off")

plt.suptitle("K-Means Segmentation", fontsize=14)
plt.tight_layout(); plt.show()

# --- Pixel Distribution per Cluster ---
from collections import Counter

plt.figure(figsize=(15, 5))
plt.suptitle("Pixel Distribution per Cluster", fontsize=14)
for i, k in enumerate(k_values):
    _, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = Counter(labels.flatten())
    idx    = sorted(counts.keys())
    clr    = centers[idx] / 255.0

    plt.subplot(1, 3, i + 1)
    plt.bar(range(k), [counts[j] for j in idx], color=clr)
    plt.title(f"K = {k}"); plt.xlabel("Cluster"); plt.ylabel("Pixel Count")
    plt.xticks(range(k)); plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout(); plt.show()


# =============================================================================
# LAB 4 — Harris Corner Detection (from scratch)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter, maximum_filter

# --- Create Checkerboard Image ---
def create_checkerboard(size=400, tile=50):
    img = np.zeros((size, size), dtype=np.float64)
    for i in range(0, size, tile):
        for j in range(0, size, tile):
            if (i // tile + j // tile) % 2 == 0:
                img[i:i + tile, j:j + tile] = 255.0
    return img

image_gray = create_checkerboard()
plt.figure(figsize=(5, 5))
plt.imshow(image_gray, cmap="gray"); plt.title("Input (Checkerboard)"); plt.axis("off"); plt.show()

# --- Step 1: Compute Gradients ---
image_smooth = gaussian_filter(image_gray, sigma=1.0)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
sobel_y = np.array([[-1,-2,-1], [ 0, 0, 0], [ 1, 2, 1]], dtype=np.float64)

Ix = cv2.filter2D(image_smooth, -1, sobel_x)
Iy = cv2.filter2D(image_smooth, -1, sobel_y)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image_smooth, cmap="gray"); axes[0].set_title("Smoothed"); axes[0].axis("off")
axes[1].imshow(Ix, cmap="RdBu");          axes[1].set_title("Gradient Ix"); axes[1].axis("off")
axes[2].imshow(Iy, cmap="RdBu");          axes[2].set_title("Gradient Iy"); axes[2].axis("off")
plt.suptitle("Image Gradients (Sobel)"); plt.tight_layout(); plt.show()

# --- Step 2: Structure Tensor ---
Sx2 = gaussian_filter(Ix ** 2,  sigma=1.5)   # M[0,0]
Sy2 = gaussian_filter(Iy ** 2,  sigma=1.5)   # M[1,1]
Sxy = gaussian_filter(Ix * Iy,  sigma=1.5)   # M[0,1] = M[1,0]

# --- Step 3: Harris Response R = det(M) - k * trace(M)^2 ---
k       = 0.05
det_M   = Sx2 * Sy2 - Sxy ** 2
trace_M = Sx2 + Sy2
R       = det_M - k * (trace_M ** 2)

# --- Step 4: Eigenvalues (for visualization) ---
disc    = np.sqrt(np.maximum(0, (trace_M ** 2) / 4 - det_M))
lambda1 = trace_M / 2 + disc
lambda2 = trace_M / 2 - disc

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0,0].imshow(Sx2,    cmap="hot"); axes[0,0].set_title("ΣIx² (M₁₁)");      axes[0,0].axis("off")
axes[0,1].imshow(Sy2,    cmap="hot"); axes[0,1].set_title("ΣIy² (M₂₂)");      axes[0,1].axis("off")
axes[0,2].imshow(Sxy, cmap="RdBu");   axes[0,2].set_title("ΣIxIy (M₁₂)");     axes[0,2].axis("off")
axes[1,0].imshow(lambda1, cmap="hot"); axes[1,0].set_title("λ₁ (larger)");     axes[1,0].axis("off")
axes[1,1].imshow(lambda2, cmap="hot"); axes[1,1].set_title("λ₂ (smaller)");    axes[1,1].axis("off")
im = axes[1,2].imshow(R, cmap="RdYlGn"); axes[1,2].set_title("Harris R"); axes[1,2].axis("off")
plt.colorbar(im, ax=axes[1,2])
plt.suptitle("Structure Tensor, Eigenvalues & Harris Response"); plt.tight_layout(); plt.show()

# --- Step 5 & 6: Threshold + NMS ---
threshold_value = 0.01 * R.max()
corner_mask = R > threshold_value
local_max   = maximum_filter(R, size=10)
corners_nms = (R == local_max) & corner_mask
corner_y, corner_x = np.where(corners_nms)
print(f"Threshold: {threshold_value:.4f} | Corners after NMS: {len(corner_x)}")

# --- Step 7: Visualize Corners ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(corner_mask, cmap="gray");     axes[0].set_title("Thresholded R"); axes[0].axis("off")
axes[1].imshow(R, cmap="RdYlGn")
axes[1].scatter(corner_x, corner_y, c="blue", s=40, marker="x", linewidths=1.5)
axes[1].set_title("R Heatmap + Corners"); axes[1].axis("off")
axes[2].imshow(np.stack([image_gray]*3, axis=-1).astype(np.uint8))
axes[2].scatter(corner_x, corner_y, c="red", s=60, marker="+", linewidths=2)
axes[2].set_title(f"Detected Corners ({len(corner_x)})"); axes[2].axis("off")
plt.suptitle("Harris Corner Detection Results"); plt.tight_layout(); plt.show()

# --- Step 8: Effect of k ---
k_values_harris = [0.02, 0.04, 0.06, 0.10]
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, kv in zip(axes, k_values_harris):
    R_k     = det_M - kv * (trace_M ** 2)
    lmax_k  = maximum_filter(R_k, size=10)
    cy, cx  = np.where((R_k == lmax_k) & (R_k > 0.01 * R_k.max()))
    ax.imshow(image_gray, cmap="gray")
    ax.scatter(cx, cy, c="red", s=60, marker="+", linewidths=2)
    ax.set_title(f"k={kv}  ({len(cx)} corners)"); ax.axis("off")
plt.suptitle("Effect of Harris k Parameter"); plt.tight_layout(); plt.show()


# =============================================================================
# LAB 5 — Object Classification: HOG + SGD
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import warnings
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import SGDClassifier
from skimage.feature import hog
import kagglehub

# Download latest version
path = kagglehub.dataset_download("owm4096/street-objects")

print("Path to dataset files:", path)

# --- Dataset Loading & Preprocessing ---
path = "dataset"                          # replace with your dataset root folder
                                          # expected structure: dataset/classA/, dataset/classB/, ...
IMG_SIZE = 64
data, labels = [], []

for category in os.listdir(path):
    cat_path = os.path.join(path, category)
    if not os.path.isdir(cat_path):
        continue
    for img_name in os.listdir(cat_path):
        try:
            img = cv2.imread(os.path.join(cat_path, img_name))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))          # 1. Resize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         # 2. Grayscale
            blur = cv2.GaussianBlur(gray, (5, 5), 0)             # 3. Denoise
            eq   = cv2.equalizeHist(blur)                        # 4. Equalize
            norm = eq / 255.0                                    # 5. Normalize
            data.append(norm)
            labels.append(category)
        except:
            pass

data   = np.array(data)
labels = np.array(labels)
print("Dataset shape:", data.shape)

# --- Show Sample Images ---
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(data[i], cmap="gray"); plt.title(labels[i]); plt.axis("off")
plt.suptitle("Sample Images (preprocessed)"); plt.show()

# --- HOG Feature Extraction ---
features = []
for img in data:
    feat = hog(
        img,
        orientations=9,        # 9 bins → 20° per bin
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    features.append(feat)

features = np.array(features)
print("Feature shape:", features.shape)   # (N_samples, HOG_dim)

# --- Train / Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# --- Train SGD Classifier ---
model = SGDClassifier(loss="log_loss", max_iter=1000)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout(); plt.show()

# --- Loss over Iterations ---
warnings.filterwarnings("ignore")
loss_values = []
model_iter = SGDClassifier(loss="log_loss", max_iter=1, warm_start=True)
for i in range(50):
    model_iter.fit(X_train, y_train)
    y_p  = model_iter.predict(X_test)
    loss_values.append(1 - accuracy_score(y_test, y_p))

plt.figure()
plt.plot(loss_values, color="steelblue", linewidth=2)
plt.title("Loss vs Iterations (SGD)"); plt.xlabel("Iteration"); plt.ylabel("Loss (1 - Accuracy)")
plt.grid(True, alpha=0.4); plt.tight_layout(); plt.show()