import pickle
from jax import jit
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
import h5py

# Define the multi-output neural network model
class MultiOutputNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(16)(x)
        x = nn.relu(x)
        
        # Separate paths for each target metric
        #voc = nn.Dense(8)(x)
        #voc = nn.relu(voc)
        voc = nn.relu(x)
        voc = nn.Dense(1)(voc)

        #jsc = nn.Dense(8)(x)
        #jsc = nn.relu(jsc)
        jsc = nn.relu(x)
        jsc = nn.Dense(1)(jsc)

        #ff = nn.Dense(8)(x)
        ff = nn.relu(x)
        #ff = nn.relu(ff)
        ff = nn.Dense(1)(ff)

        # Concatenate the outputs
        return jnp.concatenate([voc, jsc, ff], axis=-1)


# Calculate metrics function
def calculate_metrics(voltage, current):
    # Create a boolean mask for the first quadrant (V > 0 and J > 0)
    mask = (voltage > 0) & (-current > 0)

    # Apply the mask to filter V and J
    voltage = voltage[mask]
    current = -current[mask]
    voc = voltage[np.argmin(np.abs(current))]  # Voc: Voltage where current is 0
    jsc = current[np.argmin(np.abs(voltage))]  # Jsc: Current where voltage is 0
    
    # Calculate power
    power = voltage * current

    # Find the index of the maximum power point
    idx_mpp = np.argmax(power)

    # Extract Vmpp and Jmpp
    vmpp = voltage[idx_mpp]
    jmpp = current[idx_mpp]
    
    ff = (jmpp*vmpp)/(jsc*voc)
    
    return voc, jsc, ff
    

# Define a function to load and process data from multiple files
def load_data_with_thickness(filenames, thickness_values):
    """
    Load data from multiple files and add thickness as an input parameter.

    Parameters:
    filenames: List of file names to load.
    thickness_values: List of thickness values corresponding to each file.

    Returns:
    X_combined: Combined input data with thickness added as a feature.
    y_combined: Combined target metrics (Voc, Jsc, FF).
    vol_swp: Voltage sweep array (assumed to be the same for all files).
    """
    X_combined = []
    y_combined = []
    vol_swp = None

    for filename, thickness in zip(filenames, thickness_values):
        with h5py.File(filename, "r") as f:
            par_mat = np.array(f['par_mat'])             # Input parameters (6 features)
            jv_sim = np.array(f['jv_sim'])               # Output JV curves
            if vol_swp is None:
                vol_swp = np.array(f['vol_swp'])         # Voltage sweep (shared across files)

            # Add thickness as an additional feature
            thickness_column = np.full((par_mat.shape[0], 1), thickness)
            par_mat_with_thickness = np.hstack([par_mat, thickness_column])

            # Calculate the target metrics (Voc, Jsc, FF)
            target_metrics = np.array([calculate_metrics(vol_swp, jv) for jv in jv_sim])

            # Append to the combined data
            X_combined.append(par_mat_with_thickness)
            y_combined.append(target_metrics)

    # Concatenate data from all files
    X_combined = np.vstack(X_combined)
    y_combined = np.vstack(y_combined)

    return X_combined, y_combined, vol_swp


# Save the model parameters and scaler objects to a file
def save_model(filename, params, scaler_X, scaler_voc, scaler_jsc, scaler_ff):
    with open(filename, 'wb') as f:
        pickle.dump({
            'params': params,
            'scaler_X': scaler_X,
            'scaler_voc': scaler_voc,
            'scaler_jsc': scaler_jsc,
            'scaler_ff': scaler_ff
        }, f)

# Load the model parameters and scaler objects from the file
def load_model(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['params'], data['scaler_X'], data['scaler_voc'], data['scaler_jsc'], data['scaler_ff']


