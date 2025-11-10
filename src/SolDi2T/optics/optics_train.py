import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from SolDi2T.optics.interpolate_absorption import AbsorptionSurfaceModel, save_absorption_surface_model
# ----------------------------
# Utils
# ----------------------------

def correct_thickness(val):
    return round(round(val * 4) / 4, 2)

def extract_thickness(filename):
    match = re.search(r"([\d.]+)nm", filename)
    if match:
        return float(match.group(1))
    return None

# ----------------------------
# Load Data
# ----------------------------

data_dir = "data/"
all_data = []

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        raw_thickness = extract_thickness(filename)
        if raw_thickness is None:
            continue
        corrected_thickness = correct_thickness(raw_thickness)

        absorption_array = np.load(os.path.join(data_dir, filename))  # shape (1, 90, 901)
        absorption_array = absorption_array.squeeze()  # shape (90, 901)

        angles = np.arange(90)  # 0 to 89
        wavelengths = np.linspace(300, 1200, 901)  # 300 to 1200 nm

        angle_grid, wavelength_grid = np.meshgrid(angles, wavelengths, indexing='ij')

        df = pd.DataFrame({
            "thickness": corrected_thickness,
            "angle": angle_grid.flatten(),
            "wavelength": wavelength_grid.flatten(),
            "absorption": absorption_array.flatten()
        })

        all_data.append(df)

full_df = pd.concat(all_data, ignore_index=True)
full_df = full_df.sort_values(by=["thickness", "angle", "wavelength"]).reset_index(drop=True)

# ----------------------------
# Build Dataset: thickness -> (90, 901) absorption
# ----------------------------

X = []
y = []

unique_thicknesses = full_df["thickness"].unique()

for thickness in unique_thicknesses:
    subset = full_df[full_df["thickness"] == thickness]
    absorption_array = subset.sort_values(["angle", "wavelength"])["absorption"].values
    absorption_array = absorption_array.reshape(90, 901)
    X.append([thickness])
    y.append(absorption_array)

X = np.array(X, dtype=np.float32)  # (n_samples, 1)
y = np.array(y, dtype=np.float32)  # (n_samples, 90, 901)

# ----------------------------
# Train/Test Split and Scaling
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

X_train_jax = jnp.array(X_train_scaled)
X_test_jax = jnp.array(X_test_scaled)
y_train_jax = jnp.array(y_train)
y_test_jax = jnp.array(y_test)

# ----------------------------
# Model Definition
# ----------------------------

model = AbsorptionSurfaceModel()
key = jax.random.PRNGKey(0)
params = model.init(key, X_train_jax[:1])['params']

# ----------------------------
# Training Setup
# ----------------------------

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

def loss_fn(params, x, y):
    preds = model.apply({'params': params}, x)
    return jnp.mean((preds - y) ** 2)

@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# ----------------------------
# Training Loop
# ----------------------------

batch_size = 32
epochs = 50
n_train = X_train_jax.shape[0]

train_losses = []

for epoch in range(epochs):
    perm = jax.random.permutation(key, n_train)
    epoch_loss = []

    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        x_batch = X_train_jax[idx]
        y_batch = y_train_jax[idx]
        params, opt_state, loss = train_step(params, opt_state, x_batch, y_batch)
        epoch_loss.append(loss)

    mean_loss = jnp.mean(jnp.array(epoch_loss))
    train_losses.append(mean_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.6f}")

# ----------------------------
# Plot Training Loss
# ----------------------------

plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.xlim([0,10])
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Evaluation & Plot
# ----------------------------

y_pred = model.apply({'params': params}, X_test_jax)
y_pred = np.array(y_pred)


#save_absorption_surface_model(model, params, scaler_X, 'absorption_surface_model.pkl')
