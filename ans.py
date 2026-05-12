import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# PRACTICAL 1 — Matplotlib Basic Plots
# =============================================================================

x = np.linspace(0, 10, 100)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1); plt.plot(x, np.sin(x)); plt.title("Line Plot")
plt.subplot(2, 3, 2); plt.scatter(x[::5], np.sin(x[::5])); plt.title("Scatter")
plt.subplot(2, 3, 3); plt.bar(['A','B','C','D'], [3,7,2,5]); plt.title("Bar")
plt.subplot(2, 3, 4); plt.hist(np.random.randn(1000), bins=30); plt.title("Histogram")
plt.subplot(2, 3, 5); plt.pie([30,25,20,25], labels=['A','B','C','D'], autopct='%1.1f%%'); plt.title("Pie")
plt.subplot(2, 3, 6); plt.plot(x, np.sin(x), label='sin'); plt.plot(x, np.cos(x), label='cos'); plt.legend(); plt.title("Multi-line")

plt.tight_layout(); plt.show()


# =============================================================================
# PRACTICAL 2 — NumPy Matrix Operations
# =============================================================================

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("Add:\n",        A + B)
print("Sub:\n",        A - B)
print("Mul (@ ):\n",   A @ B)        # matrix multiply — NOT A*B (element-wise)
print("Transpose:\n",  A.T)
print("Det:",          np.linalg.det(A))
print("Inv:\n",        np.linalg.inv(A))
print("A@inv(A):\n",   np.round(A @ np.linalg.inv(A)))   # should be Identity


# =============================================================================
# PRACTICAL 3 — Activation Functions
# =============================================================================

x = np.linspace(-5, 5, 200)

def sigmoid(x):    return 1 / (1 + np.exp(-x))
def tanh(x):       return np.tanh(x)
def relu(x):       return np.maximum(0, x)
def leaky_relu(x): return np.where(x > 0, x, 0.01 * x)
def linear(x):     return x

fns   = [sigmoid, tanh, relu, leaky_relu, linear]
names = ['Sigmoid', 'Tanh', 'ReLU', 'Leaky ReLU', 'Linear']

plt.figure(figsize=(12, 4))
for i, (fn, name) in enumerate(zip(fns, names)):
    plt.subplot(1, 5, i+1)
    plt.plot(x, fn(x)); plt.title(name)
    plt.axhline(0, color='k', lw=0.5); plt.axvline(0, color='k', lw=0.5)
plt.tight_layout(); plt.show()


# =============================================================================
# PRACTICAL 4 — Delta Learning Rule (Weight & Bias Effect)
# =============================================================================

def train_delta(X, y, lr=0.01, epochs=100):
    w = np.zeros(X.shape[1]); b = 0.0; errors = []
    for _ in range(epochs):
        total_err = 0
        for xi, yi in zip(X, y):
            out    = np.dot(xi, w) + b
            err    = yi - out
            w     += lr * err * xi   # delta rule
            b     += lr * err
            total_err += err**2
        errors.append(total_err)
    return w, b, errors

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([0, 1, 1, 1], dtype=float)   # OR gate

w, b, errors = train_delta(X, y)
print("Weights:", w, "Bias:", b)
plt.plot(errors); plt.title("Delta Rule — Error"); plt.xlabel("Epoch"); plt.show()
for xi in X:
    print(xi, "->", round(np.dot(xi, w) + b))


# =============================================================================
# PRACTICAL 5 — Gradient Descent to Minimise Error (ANN Optimisation)
# =============================================================================

np.random.seed(42)
X_gd = 2 * np.random.rand(100, 1)
y_gd = 3 * X_gd.squeeze() + 1 + np.random.randn(100) * 0.5   # y = 3x+1+noise

X_b  = np.c_[np.ones(100), X_gd]   # add bias column
w_gd = np.zeros(2)
lr   = 0.05; losses = []

for epoch in range(200):
    y_pred = X_b @ w_gd
    error  = y_pred - y_gd
    loss   = np.mean(error**2)
    grad   = (2 / len(X_gd)) * X_b.T @ error   # dL/dw = (2/N) * Xᵀ(Xw-y)
    w_gd  -= lr * grad
    losses.append(loss)

print(f"GD learned — bias: {w_gd[0]:.3f}, weight: {w_gd[1]:.3f}")   # ~1, ~3
plt.plot(losses); plt.title("GD Loss"); plt.xlabel("Epoch"); plt.show()


# =============================================================================
# PRACTICAL 6 — OR, AND, XOR Gates (Linear vs Non-linearly Separable)
# =============================================================================

def step(x): return 1 if x >= 0 else 0

def perceptron_train(X, y, lr=0.1, epochs=100):
    w = np.zeros(2); b = 0.0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            pred = step(np.dot(xi, w) + b)
            err  = yi - pred
            w   += lr * err * xi
            b   += lr * err
    return w, b

X_g = np.array([[0,0],[0,1],[1,0],[1,1]])
gates = {
    'OR':  np.array([0,1,1,1]),
    'AND': np.array([0,0,0,1]),
    'XOR': np.array([0,1,1,0]),   # NOT linearly separable — will fail
}

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, yg) in zip(axes, gates.items()):
    w, b = perceptron_train(X_g, yg)
    preds = [step(np.dot(xi, w) + b) for xi in X_g]
    print(f"{name}: preds={preds}, targets={list(yg)}, match={preds == list(yg)}")

    for xi, yi in zip(X_g, yg):
        ax.scatter(*xi, c='blue' if yi == 1 else 'red', s=200, zorder=3)
        ax.annotate(f"({int(xi[0])},{int(xi[1])})", xi, textcoords="offset points", xytext=(5,5), fontsize=8)

    if w[1] != 0:
        xs = np.linspace(-0.5, 1.5, 100)
        ax.plot(xs, -(w[0]*xs + b)/w[1], 'g--', label='Boundary')
    else:
        ax.axvline(-b/w[0] if w[0] != 0 else 0.5, color='green', linestyle='--', label='Boundary')

    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_title(f"{name} Gate"); ax.legend(fontsize=7)

plt.suptitle("Gate Decision Boundaries — XOR has no valid line", y=1.02)
plt.tight_layout(); plt.show()
# OR -> True, AND -> True, XOR -> False (needs hidden layer)


# =============================================================================
# PRACTICAL 7 — Backpropagation ANN (solves XOR)
# =============================================================================

def sigmoid(x):    return 1 / (1 + np.exp(-x))
def sigmoid_d(a):  return a * (1 - a)   # a is already sigmoid output

X_bp = np.array([[0,0],[0,1],[1,0],[1,1]])
y_bp = np.array([[0],[1],[1],[0]])          # XOR

np.random.seed(42)
W1 = np.random.randn(2, 2); b1 = np.zeros((1, 2))   # 2->2 hidden
W2 = np.random.randn(2, 1); b2 = np.zeros((1, 1))   # 2->1 output
lr = 0.5; bp_losses = []

for epoch in range(10000):
    a1 = sigmoid(X_bp @ W1 + b1)
    a2 = sigmoid(a1   @ W2 + b2)

    d2  = -(y_bp - a2) * sigmoid_d(a2)
    dW2 = a1.T @ d2;   db2 = np.sum(d2, axis=0, keepdims=True)
    d1  = (d2 @ W2.T) * sigmoid_d(a1)
    dW1 = X_bp.T @ d1; db1 = np.sum(d1, axis=0, keepdims=True)

    W2 -= lr*dW2; b2 -= lr*db2
    W1 -= lr*dW1; b1 -= lr*db1
    bp_losses.append(np.mean((y_bp - a2)**2))

print("Backprop XOR predictions:\n", np.round(a2))   # [[0],[1],[1],[0]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(bp_losses, color='steelblue')
ax1.set_title("Backprop — Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("MSE")

x_pos = np.arange(4); width = 0.35
ax2.bar(x_pos - width/2, y_bp.flatten(), width, label='Target',    color='steelblue')
ax2.bar(x_pos + width/2, a2.flatten(),   width, label='Predicted', color='coral')
ax2.set_xticks(x_pos); ax2.set_xticklabels(['(0,0)','(0,1)','(1,0)','(1,1)'])
ax2.set_ylim(0, 1.3); ax2.set_title("XOR Predictions vs Targets"); ax2.legend()
plt.tight_layout(); plt.show()


# =============================================================================
# PRACTICAL 8 — ADALINE with LMS Algorithm
# =============================================================================

# ADALINE differs from Perceptron: update uses RAW linear output, not step(output)

X_ad = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_ad = np.array([-1, 1, 1, 1], dtype=float)   # bipolar OR targets

w_ad = np.zeros(2); b_ad = 0.0; lr = 0.01; lms_losses = []

for epoch in range(200):
    total = 0
    for xi, yi in zip(X_ad, y_ad):
        out    = np.dot(xi, w_ad) + b_ad   # LINEAR — no activation for update
        err    = yi - out
        w_ad  += lr * err * xi             # LMS / Widrow-Hoff rule
        b_ad  += lr * err
        total += err**2
    lms_losses.append(total)

plt.plot(lms_losses); plt.title("ADALINE LMS Loss"); plt.show()

def sign(x): return 1 if x >= 0 else -1
for xi in X_ad:
    print(xi, "->", sign(np.dot(xi, w_ad) + b_ad))


# =============================================================================
# PRACTICAL 9 — Self-Organising Map (SOM)
# =============================================================================

np.random.seed(42)
data = np.random.rand(100, 2)          # 100 random 2D points

grid  = 5
W_som = np.random.rand(grid, grid, 2)  # 5x5 grid, each node = 2D weight

def find_bmu(x, W):
    return np.unravel_index(np.argmin(np.linalg.norm(W - x, axis=2)), (grid, grid))

def h(bmu, i, j, sigma):
    return np.exp(-((bmu[0]-i)**2 + (bmu[1]-j)**2) / (2 * sigma**2))

lr0 = 0.5; sigma0 = 2.0; epochs = 500

for t in range(epochs):
    lr    = lr0    * np.exp(-t / epochs)
    sigma = sigma0 * np.exp(-t / epochs)
    x     = data[np.random.randint(len(data))]
    bmu   = find_bmu(x, W_som)
    for i in range(grid):
        for j in range(grid):
            W_som[i, j] += lr * h(bmu, i, j, sigma) * (x - W_som[i, j])

plt.figure(figsize=(6, 6))
plt.scatter(data[:,0], data[:,1], alpha=0.4, label='Data')
plt.scatter(W_som[:,:,0].flatten(), W_som[:,:,1].flatten(),
            c='red', marker='x', s=100, label='SOM nodes')
plt.title("SOM Clustering"); plt.legend(); plt.show()
