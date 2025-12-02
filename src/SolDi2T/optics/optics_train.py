import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ----------------------------------------
# Path Setup
# ----------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from SolDi2T.optics.interpolate_absorption import AbsorptionSurfaceModel, save_absorption_surface_model

DATA_DIR = "data/absorption/npy"
SAVE_DIR = "data/absorption"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
fontsize = 18

# ----------------------------------------
# Utilities
# ----------------------------------------
def correct_thickness(val: float):
    return round(round(val * 4) / 4, 2)

def extract_thickness(filename: str):
    match = re.search(r"([\d.]+)nm", filename)
    return float(match.group(1)) if match else None

# ----------------------------------------
# Load Data
# ----------------------------------------
all_dfs = []
angles = np.arange(90)
wavelengths = np.linspace(300, 1200, 901)

angle_grid, wavelength_grid = np.meshgrid(angles, wavelengths, indexing="ij")

for filename in os.listdir(DATA_DIR):
    if not filename.endswith(".npy"):
        continue

    raw_t = extract_thickness(filename)
    if raw_t is None:
        continue

    t = correct_thickness(raw_t)
    arr = np.load(os.path.join(DATA_DIR, filename)).squeeze()  # shape (90, 901)

    df = pd.DataFrame({
        "thickness": t,
        "angle": angle_grid.flatten(),
        "wavelength": wavelength_grid.flatten(),
        "absorption": arr.flatten()
    })
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = full_df.sort_values(["thickness", "angle", "wavelength"]).reset_index(drop=True)

# ----------------------------------------
# Build ML Dataset
# ----------------------------------------
X, y = [], []
for t in full_df["thickness"].unique():
    sub = full_df[full_df["thickness"] == t]
    arr = sub.sort_values(["angle", "wavelength"])["absorption"].values.reshape(90, 901)
    X.append([t])
    y.append(arr)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# ----------------------------------------
# Train/Test Split & Standardization
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)

X_train_jax = jnp.array(X_train_s)
X_test_jax = jnp.array(X_test_s)
y_train_jax = jnp.array(y_train)
y_test_jax = jnp.array(y_test)

# ----------------------------------------
# Model and Optimizer
# ----------------------------------------
model = AbsorptionSurfaceModel()
key = jax.random.PRNGKey(0)
params = model.init(key, X_train_jax[:1])["params"]

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def loss_fn(p, x, y):
    preds = model.apply({"params": p}, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(p, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    p = optax.apply_updates(p, updates)
    return p, opt_state, loss

# ----------------------------------------
# Training Loop
# ----------------------------------------
epochs = 10
batch_size = 32
n_train = len(X_train_jax)
train_losses = []

for epoch in range(epochs):
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_train)

    batch_losses = []
    for start in range(0, n_train, batch_size):
        idx = perm[start:start + batch_size]
        xb, yb = X_train_jax[idx], y_train_jax[idx]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        batch_losses.append(loss)

    mean_loss = float(jnp.mean(jnp.array(batch_losses)))
    train_losses.append(mean_loss)

    if epoch % 2 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {mean_loss:.6f}")

# ----------------------------------------
# Plot Training Loss
# ----------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("MSE Loss", fontsize=fontsize)
#plt.title("Training Loss Over Epochs", fontsize=fontsize)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/traininglossoptics.png", dpi=300)
plt.close()

# ----------------------------------------
# Predictions
# ----------------------------------------
y_pred = model.apply({"params": params}, X_test_jax)
y_pred = np.array(y_pred)

# ----------------------------------------
# KDE Density Plot
# ----------------------------------------
y_true_flat = y_test.reshape(-1)
y_pred_flat = y_pred.reshape(-1)

N = min(500000, len(y_true_flat))
idx = np.random.choice(len(y_true_flat), size=N, replace=False)

plt.figure(figsize=(7, 6))
sns.kdeplot(x=y_true_flat[idx], y=y_pred_flat[idx], fill=True, thresh=0.05)
plt.plot([0, 1], [0, 1], "r--", lw=2)
plt.xlabel("True Generation Rates", fontsize=fontsize)
plt.ylabel("Predicted Generation Rates", fontsize=fontsize)
#plt.title("True vs Predicted Generation Rates", fontsize=fontsize)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/predictedvstrueplotdensity.png", dpi=300)
plt.close()

# ----------------------------------------
# Example Visualization
# ----------------------------------------
i = 24
fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

im = axs[0].imshow(y_pred[i], aspect='auto', extent=[300,1200,89,0], cmap='viridis')
axs[0].set_title(f"Predicted Generation Rates\nThickness = {X_test[i][0]:.2f} nm", fontsize=fontsize)
axs[0].set_xlabel("Wavelength [nm]", fontsize=fontsize)
axs[0].set_ylabel("Angle [deg]", fontsize=fontsize)
fig.colorbar(im, ax=axs[0], ticks=[0, 0.2, 0.4, 0.6, 0.8])

im = axs[1].imshow(y_test[i], aspect='auto', extent=[300,1200,89,0], cmap='viridis')
axs[1].set_title(f"True Generation Rates\nThickness = {X_test[i][0]:.2f} nm", fontsize=fontsize)
axs[1].set_xlabel("Wavelength [nm]", fontsize=fontsize)
axs[1].set_ylabel("Angle [deg]", fontsize=fontsize)
fig.colorbar(im, ax=axs[1], ticks=[0, 0.2, 0.4, 0.6, 0.8])

#plt.suptitle("Predicted vs True Generation Rates", fontsize=24)
plt.savefig(f"{SAVE_DIR}/predictedvcalculatedabsorption.png", dpi=300)
plt.close()

# ----------------------------------------
# Optional: Save model
# ----------------------------------------
# save_absorption_surface_model(model, params, scaler_X, "absorption_surface_model.pkl")
