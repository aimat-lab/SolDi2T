import jax.numpy as jnp
from jax import jit
from interpax import interp1d
from flax import linen as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class AbsorptionSurfaceModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(128)(x))
        x = nn.relu(nn.Dense(256)(x))
        x = nn.Dense(90 * 901)(x)
        return x.reshape((-1, 90, 901))
    
def save_absorption_surface_model(model, params, scaler_X, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump({
            'model': model,
            'params': params,
            'scaler_X': scaler_X
        }, f)


def load_absorption_surface_model(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['params'], data['scaler_X']

def predict_absorption_surface_NN(thickness_value, model, params, scaler_X):
    # Convert to JAX array
    thickness_jax = jnp.array([[thickness_value]], dtype=jnp.float32)  # shape (1, 1)

    # Extract scaler parameters for manual standardization
    mean_X = jnp.array(scaler_X.mean_, dtype=jnp.float32)
    std_X = jnp.array(scaler_X.scale_, dtype=jnp.float32)

    # Scale thickness manually
    thickness_scaled = (thickness_jax - mean_X) / std_X  # shape (1, 1)

    # Predict absorption surface
    prediction = model.apply({'params': params}, thickness_scaled)  # shape (1, 90, 901)

    return prediction

