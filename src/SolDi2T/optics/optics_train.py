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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from SolDi2T.optics.interpolate_absorption import AbsorptionSurfaceModel, save_absorption_surface_model

# ----------------------------------------
# Path Setup & Utilities
# ----------------------------------------
DATA_DIR = "data/absorption/npy"
SAVE_DIR = "final_results/optics"
os.makedirs(SAVE_DIR, exist_ok=True)

plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
fontsize = 18

def save_plot(filename_base):
    plt.savefig(f"{SAVE_DIR}/{filename_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{SAVE_DIR}/{filename_base}.pdf", bbox_inches='tight')
    plt.close()

def correct_thickness(val: float):
    return round(round(val * 4) / 4, 2)

def extract_thickness(filename: str):
    match = re.search(r"([\d.]+)nm", filename)
    return float(match.group(1)) if match else None

# Load Data
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
    arr = np.load(os.path.join(DATA_DIR, filename)).squeeze() 
    df = pd.DataFrame({
        "thickness": t, "angle": angle_grid.flatten(),
        "wavelength": wavelength_grid.flatten(), "absorption": arr.flatten()
    })
    all_dfs.append(df)

full_df = pd.concat(all_dfs, ignore_index=True)
full_df = full_df.sort_values(["thickness", "angle", "wavelength"]).reset_index(drop=True)

X, y = [], []
for t in full_df["thickness"].unique():
    sub = full_df[full_df["thickness"] == t]
    arr = sub.sort_values(["angle", "wavelength"])["absorption"].values.reshape(90, 901)
    X.append([t])
    y.append(arr)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Train/Val/Test Split & Scaling
def stratified_split(mask):
    Xm, ym = X[mask], y[mask]
    Xt, Xte, yt, yte = train_test_split(Xm, ym, test_size=0.2, random_state=42)
    Xtr, Xv, ytr, yv = train_test_split(Xt, yt, test_size=0.2, random_state=42)
    return (Xtr, ytr), (Xv, yv), (Xte, yte)

thin = X[:, 0] < 100
parts = [stratified_split(thin), stratified_split(~thin)]
X_train, y_train = (np.concatenate([p[0][i] for p in parts]) for i in (0, 1))
X_val, y_val = (np.concatenate([p[1][i] for p in parts]) for i in (0, 1))
X_test, y_test = (np.concatenate([p[2][i] for p in parts]) for i in (0, 1))
print(f"Train/val/test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)} "
      f"(thin <100nm: {int((X_train[:,0]<100).sum())}/{int((X_val[:,0]<100).sum())}/{int((X_test[:,0]<100).sum())})")

scaler_X = StandardScaler()
X_train_jax = jnp.array(scaler_X.fit_transform(X_train))
X_val_jax = jnp.array(scaler_X.transform(X_val))
X_test_jax = jnp.array(scaler_X.transform(X_test))

y_train_jax = jnp.array(y_train)
y_val_jax = jnp.array(y_val)
y_test_jax = jnp.array(y_test)

# ---------------------------------------------------------
# Batch Optimization & Early Stopping
# ---------------------------------------------------------
model = AbsorptionSurfaceModel()
key = jax.random.PRNGKey(0)

learning_rate = 1e-3
max_epochs = 200
patience = 15
batch_sizes_to_test = [8, 16, 32]

best_global_val_loss = float('inf')
best_global_params = None
best_global_batch_size = None
best_train_losses_history = []
best_val_losses_history = []

def loss_fn(p, x, y):
    preds = model.apply({"params": p}, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(p, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(p, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    p = optax.apply_updates(p, updates)
    return p, opt_state, loss

print("\n--- Starting Batch Optimization ---")

for bs in batch_sizes_to_test:
    print(f"\nTesting Batch Size: {bs}")
    params = model.init(key, X_train_jax[:1])["params"]
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    n_train = len(X_train_jax)
    
    best_bs_val_loss = float('inf')
    best_bs_params = None
    patience_counter = 0
    
    bs_train_losses = []
    bs_val_losses = []

    for epoch in range(max_epochs):
        key, subkey = jax.random.split(key)
        perm = jax.random.permutation(subkey, n_train)

        epoch_losses = []
        for start in range(0, n_train, bs):
            idx = perm[start:start + bs]
            xb, yb = X_train_jax[idx], y_train_jax[idx]
            params, opt_state, loss = train_step(params, opt_state, xb, yb)
            epoch_losses.append(loss)

        mean_train_loss = float(jnp.mean(jnp.array(epoch_losses)))
        val_loss = float(loss_fn(params, X_val_jax, y_val_jax))
        
        bs_train_losses.append(mean_train_loss)
        bs_val_losses.append(val_loss)

        if val_loss < best_bs_val_loss - 1e-5:
            best_bs_val_loss = val_loss
            best_bs_params = params
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  Early stopping triggered at epoch {epoch+1}. Best Val Loss: {best_bs_val_loss:.6f}")
            break

    if best_bs_val_loss < best_global_val_loss:
        best_global_epochs = int(np.argmin(bs_val_losses)) + 1
        best_global_val_loss = best_bs_val_loss
        best_global_params = best_bs_params
        best_global_batch_size = bs
        best_train_losses_history = bs_train_losses
        best_val_losses_history = bs_val_losses

print(f"\nOptimization Complete! Best Batch Size selected: {best_global_batch_size}")

# ---------------------------------------------------------
# Final Evaluation & Plots
# ---------------------------------------------------------
print("\nEvaluating on Holdout Test Set...")
y_pred_test = model.apply({"params": best_global_params}, X_test_jax)
y_pred_test_np = np.array(y_pred_test)

y_test_flat = y_test.flatten()
y_pred_flat = y_pred_test_np.flatten()

rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
mae = mean_absolute_error(y_test_flat, y_pred_flat)
r2 = r2_score(y_test_flat, y_pred_flat)

print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE:  {mae:.6f}")
print(f"Test R^2:  {r2:.6f}")

# Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(best_train_losses_history)+1), best_train_losses_history, marker='o', label='Training Loss')
plt.plot(range(1, len(best_val_losses_history)+1), best_val_losses_history, marker='s', label='Validation Loss')
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("MSE Loss", fontsize=fontsize)
plt.xlim([0, 10])
plt.legend(fontsize=14)
plt.grid(True)
save_plot(f"training_validation_loss_optics_bs{best_global_batch_size}")

# Density Plot
N = min(500000, len(y_test_flat))
idx = np.random.choice(len(y_test_flat), size=N, replace=False)
plt.figure(figsize=(7, 6))
sns.kdeplot(x=y_test_flat[idx], y=y_pred_flat[idx], fill=True, thresh=0.05)
plt.plot([0, 1], [0, 1], "r--", lw=2)
plt.xlabel("True Gen Rates", fontsize=fontsize)
plt.ylabel("Predicted Gen Rates", fontsize=fontsize)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.grid(True)
save_plot("predicted_vs_true_density_test")

# Heatmaps
i = min(24, len(X_test) - 1) 
fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
im = axs[0].imshow(y_pred_test_np[i], aspect='auto', extent=[300,1200,89,0], cmap='viridis')
axs[0].set_title(f"Predicted Gen Rates\nThickness = {X_test[i][0]:.2f} nm", fontsize=fontsize)
fig.colorbar(im, ax=axs[0], ticks=[0, 0.2, 0.4, 0.6, 0.8])
im = axs[1].imshow(y_test[i], aspect='auto', extent=[300,1200,89,0], cmap='viridis')
axs[1].set_title(f"True Gen Rates\nThickness = {X_test[i][0]:.2f} nm", fontsize=fontsize)
fig.colorbar(im, ax=axs[1], ticks=[0, 0.2, 0.4, 0.6, 0.8])
save_plot("predicted_vs_calculated_absorption_test")

# ---------------------------------------------------------
# Final model: retrain on 100% of the data
# ---------------------------------------------------------
print(f"\nRetraining on all {len(X)} thicknesses: batch size {best_global_batch_size}, {best_global_epochs} epochs")
final_scaler_X = StandardScaler()
X_all_jax = jnp.array(final_scaler_X.fit_transform(X))
y_all_jax = jnp.array(y)

params = model.init(key, X_all_jax[:1])["params"]
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)
n_all = len(X_all_jax)
for epoch in range(best_global_epochs):
    key, subkey = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_all)
    epoch_losses = []
    for start in range(0, n_all, best_global_batch_size):
        idx = perm[start:start + best_global_batch_size]
        params, opt_state, loss = train_step(params, opt_state, X_all_jax[idx], y_all_jax[idx])
        epoch_losses.append(loss)
    print(f"  epoch {epoch+1}/{best_global_epochs} train loss {float(jnp.mean(jnp.array(epoch_losses))):.6f}")

save_absorption_surface_model(model, params, final_scaler_X, "data/absorption/absorption_surface_model.pkl")
print("Saved data/absorption/absorption_surface_model.pkl")