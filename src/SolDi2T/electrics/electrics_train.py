import numpy as np
import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optax  # Optimizer library for JAX
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import logging
import sys
import os
# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from SolDi2T.electrics.model_utils import MultiOutputNN, load_data_with_thickness, save_model


plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
fontsize=18

# List of files and corresponding thickness values
filenames = [
    "data/electrics/102nm.h5",
    "data/electrics/147nm.h5",
    "data/electrics/201nm.h5",
    "data/electrics/246nm.h5",
    "data/electrics/300nm.h5",
]
thickness_values = [102, 147, 201, 246, 300]

learning_rate = 0.01
n_epochs = 100

# Load the combined data
X_combined, y_combined, vol_swp = load_data_with_thickness(filenames, thickness_values)

print(X_combined[0])

# Split data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Standardize the input data
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Separate scaling for each target metric
scaler_voc = StandardScaler()
scaler_jsc = StandardScaler()
scaler_ff = StandardScaler()

y_train_voc_scaled = scaler_voc.fit_transform(y_train[:, [0]])
y_train_jsc_scaled = scaler_jsc.fit_transform(y_train[:, [1]])
y_train_ff_scaled = scaler_ff.fit_transform(y_train[:, [2]])
y_train_scaled = np.concatenate([y_train_voc_scaled, y_train_jsc_scaled, y_train_ff_scaled], axis=1)

y_test_voc_scaled = scaler_voc.transform(y_test[:, [0]])
y_test_jsc_scaled = scaler_jsc.transform(y_test[:, [1]])
y_test_ff_scaled = scaler_ff.transform(y_test[:, [2]])
y_test_scaled = np.concatenate([y_test_voc_scaled, y_test_jsc_scaled, y_test_ff_scaled], axis=1)

# Convert data to JAX arrays
X_train_scaled = jnp.array(X_train_scaled)
X_test_scaled = jnp.array(X_test_scaled)
y_train_scaled = jnp.array(y_train_scaled)
y_test_scaled = jnp.array(y_test_scaled)

# Initialize the model and optimizer
model = MultiOutputNN()
key = random.PRNGKey(0)
params = model.init(key, X_train_scaled)  # Initialize model parameters

# Define the loss function (Mean Squared Error)
def loss_fn(params, X, y):
    preds = model.apply(params, X)
    return jnp.mean((preds - y) ** 2)

# Set up the optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Define the training step
@jit
def train_step(params, opt_state, X, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# Train the model and record both training and validation losses
train_losses = []
val_losses = []

for epoch in range(n_epochs):
    # Perform a training step
    params, opt_state, train_loss = train_step(params, opt_state, X_train_scaled, y_train_scaled)
    
    # Calculate validation loss
    val_loss = loss_fn(params, X_test_scaled, y_test_scaled)
    
    # Store the losses
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Print training progress every 100 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

logging.info("Model trained successfully.")

# Make predictions on the test set
y_pred_scaled = model.apply(params, X_test_scaled)

# Inverse transform predictions for each metric
y_pred_voc = scaler_voc.inverse_transform(y_pred_scaled[:, [0]])
y_pred_jsc = scaler_jsc.inverse_transform(y_pred_scaled[:, [1]])
y_pred_ff = scaler_ff.inverse_transform(y_pred_scaled[:, [2]])
y_pred = np.concatenate([y_pred_voc, y_pred_jsc, y_pred_ff], axis=1)

# Similarly inverse transform the test set for evaluation
y_test_voc = scaler_voc.inverse_transform(y_test_scaled[:, [0]])
y_test_jsc = scaler_jsc.inverse_transform(y_test_scaled[:, [1]])
y_test_ff = scaler_ff.inverse_transform(y_test_scaled[:, [2]])
y_test = np.concatenate([y_test_voc, y_test_jsc, y_test_ff], axis=1)

# Evaluate the model using KDE plots
metrics_names = ['Voc [V]', 'Jsc', 'FF']


# Save the model to a file
#save_model('data/electrics/drift_diffusion_nn_model.pkl', params, scaler_X, scaler_voc, scaler_jsc, scaler_ff)
logging.info("Model saved successfully.")


# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(n_epochs), train_losses, label="Training Loss")
plt.plot(range(n_epochs), val_losses, label="Validation Loss")
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Loss", fontsize=fontsize)
#plt.title("Training and Validation Loss over Epochs")
plt.legend(fontsize=fontsize)
# Save loss curve
plt.savefig("data/electrics/elec_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

for i, name in enumerate(metrics_names):
    real = y_test[:, i]
    pred = y_pred[:, i]
    r2 = r2_score(real, pred)

    plt.figure(figsize=(5, 5))
    sns.kdeplot(x=real, y=pred, fill=True, cmap="Blues", thresh=0.05)
    plt.plot([real.min(), real.max()], [real.min(), real.max()], 'k--', lw=2)
    #plt.title(f"{name}: Real vs Predicted\nR² = {r2:.2f}")
    plt.xlabel(f"Real {name}", fontsize=fontsize)
    plt.ylabel(f"Predicted {name}", fontsize=fontsize)
    plt.axis('square')

    # Save each figure
    plt.savefig(f"data/electrics/{name.lower()}_pred_vs_true.png",
                dpi=300, bbox_inches="tight")
    plt.show()
