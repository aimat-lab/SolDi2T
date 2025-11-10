import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import logging
import time
import jax
import jax.numpy as jnp
import gc
# Disable JAX preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# -----------------------------
# Setup
# -----------------------------
location = "seattle"
opt_method = "gradient_ascent"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.info("Starting gradient ascent optimization script...")

# -----------------------------
# User parameters
# -----------------------------
rotation_angle = 180.0
CE_value = 0.84
NOCT_value = 48.0

thickness_range = np.arange(100.0, 320.0, 20.0)
tilt_angles = np.arange(0.0, 100.0, 10.0)

initial_lr_tilt = 10.0
decay_rate_tilt = 0.05
initial_lr_thickness = 50.0
decay_rate_thickness = 0.1
num_steps = 30
min_tilt, max_tilt = 0.0, 90.0
min_thickness, max_thickness = 100.0, 300.0

# -----------------------------
# Load irradiance and models
# -----------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from SolDi2T.energy_yield.irradiance import trim_irradiance
from SolDi2T.energy_yield.jsc_calculation import JscCalc_jax
from SolDi2T.optics.interpolate_absorption import (
    AbsorptionSurfaceModel,
    load_absorption_surface_model,
    predict_absorption_surface_NN,
)
from SolDi2T.electrics.model_utils import MultiOutputNN, load_model

# Load irradiance
with open(f"data/irradiance/{location}_irradiance_data.pkl", "rb") as file:
    irradiance = pickle.load(file)

lambda_ = jnp.arange(300, 1201, 1)
lambda_values = irradiance["Irr_spectra_clouds_wavelength"].flatten()
thetasun = irradiance["Data_TMY3"]["Data_TMY3"][:, 6]
phisun = irradiance["Data_TMY3"]["Data_TMY3"][:, 7]
IrradianceDifH = irradiance["Irr_spectra_clouds_diffuse_horizontal"]
IrradianceDirH = irradiance["Irr_spectra_clouds_direct_horizontal"]
TempAmbient = irradiance["Data_TMY3"]["Data_TMY3"][:, 13]

IrradianceDifN = IrradianceDifH / jnp.pi
IrradianceDirN = (jnp.ceil(thetasun) < 90)[:, None] * IrradianceDirH / jnp.abs(
    jnp.cos(jnp.deg2rad(thetasun))
)[:, None]

IdifN = trim_irradiance(lambda_, IrradianceDifN, lambda_values)
IdirN = trim_irradiance(lambda_, IrradianceDirN, lambda_values)

# Load models
loaded_absorption_surface_model, loaded_absorption_surface_params, loaded_absorption_surface_scaler_X = load_absorption_surface_model(
    "data/absorption/absorption_surface_model.pkl"
)
model = MultiOutputNN()
loaded_params, loaded_scaler_X, loaded_scaler_voc, loaded_scaler_jsc, loaded_scaler_ff = load_model(
    "data/electrics/drift_diffusion_nn_model.pkl"
)

# -----------------------------
# Output directory
# -----------------------------
output_dir = os.path.join("data", "results", location)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Energy yield wrapper
# -----------------------------
def EYCalc_wrapper(tilt_angle, thickness_value):
    A = predict_absorption_surface_NN(
        thickness_value,
        loaded_absorption_surface_model,
        loaded_absorption_surface_params,
        loaded_absorption_surface_scaler_X,
    )

    S = jnp.zeros(8760)
    NOCT = NOCT_value * jnp.ones(A.shape[0])
    CE = CE_value * jnp.ones(A.shape[0])

    Jsc_direct, Jsc_diffuse, Jsc, S = JscCalc_jax(
        thetasun, phisun, IdirN, IdifN, A, tilt_angle, rotation_angle, CE, S
    )

    temp_module = TempAmbient[:, None] + (NOCT - 20) / 800 * S[:, None]
    suns_values = S / 1000
    T_values = temp_module + 273.15

    instances = jnp.column_stack(
        (T_values, suns_values, jnp.full_like(suns_values, thickness_value))
    )
    instances_scaled = (instances - jnp.array(loaded_scaler_X.mean_)) / jnp.array(loaded_scaler_X.scale_)
    pred_scaled = model.apply(loaded_params, instances_scaled)
    voc_preds = pred_scaled[:, 0] * loaded_scaler_voc.scale_ + loaded_scaler_voc.mean_
    ff_preds = pred_scaled[:, 2] * loaded_scaler_ff.scale_ + loaded_scaler_ff.mean_
    jsc_values = Jsc.reshape(-1)
    power = 10 * voc_preds * ff_preds * jsc_values
    return jnp.sum(power) / 1000  # kWh/m²/a

grad_EYCalc = jax.grad(EYCalc_wrapper, argnums=(0, 1))

# -----------------------------
# Gradient ascent optimizer
# -----------------------------
def optimize_from_start(tilt_start, thickness_start):
    tilt_angle = jnp.array(tilt_start)
    thickness_value = jnp.array(thickness_start)

    tilt_history, thickness_history, power_history = [tilt_angle], [thickness_value], [EYCalc_wrapper(tilt_angle, thickness_value)]
    grad_tilt_history, grad_thickness_history = [], []

    for step in range(num_steps):
        lr_tilt = initial_lr_tilt * (1.0 - decay_rate_tilt) ** step
        lr_thickness = initial_lr_thickness * (1.0 - decay_rate_thickness) ** step

        grad_tilt, grad_thickness = grad_EYCalc(tilt_angle, thickness_value)
        grad_tilt_history.append(float(grad_tilt))
        grad_thickness_history.append(float(grad_thickness))

        tilt_angle += lr_tilt * grad_tilt
        thickness_value += lr_thickness * grad_thickness
        tilt_angle = jnp.clip(tilt_angle, min_tilt, max_tilt)
        thickness_value = jnp.clip(thickness_value, min_thickness, max_thickness)

        current_power = EYCalc_wrapper(tilt_angle, thickness_value)
        tilt_history.append(tilt_angle)
        thickness_history.append(thickness_value)
        power_history.append(current_power)

        logging.info(f"[Start ({tilt_start},{thickness_start}) | Step {step}] "
                     f"Tilt: {tilt_angle:.2f}, Thickness: {thickness_value:.2f}, "
                     f"Power: {current_power:.4f}, GradTilt: {grad_tilt:.4f}, GradThick: {grad_thickness:.4f}, "
                     f"lr_tilt: {lr_tilt:.3f}, lr_thickness: {lr_thickness:.3f}")

    return {
        "start": (tilt_start, thickness_start),
        "final": (float(tilt_angle), float(thickness_value)),
        "tilt_history": [float(t) for t in tilt_history],
        "thickness_history": [float(t) for t in thickness_history],
        "power_history": [float(p) for p in power_history],
        "grad_tilt_history": grad_tilt_history,
        "grad_thickness_history": grad_thickness_history
    }

# -----------------------------
# Run optimization from multiple starting points
# -----------------------------
start_points = [(10.0, 125.0), (10.0, 275.0), (50.0, 125.0), (50.0, 275.0)]
#start_points = [(10.0, 125.0), (50.0, 275.0)]
all_results = []
start_time = time.time()
for tilt, thick in start_points:
    logging.info(f"Starting optimization from Tilt: {tilt}, Thickness: {thick}")
    result = optimize_from_start(tilt, thick)
    all_results.append(result)
end_time = time.time()
logging.info(f"Gradient ascent optimization finished in {end_time - start_time:.2f} seconds")

# Save optimization results
results_path = f"{output_dir}/gradient_ascent_results.pkl"
with open(results_path, "wb") as f:
    pickle.dump(all_results, f)
logging.info(f"Saved gradient ascent results to {results_path}")

# -----------------------------
# Plot trajectories (same as your previous plotting code)
# -----------------------------
# Prepare DataFrame of trajectories
trajectories = []
for res in all_results:
    start_tilt, start_thick = res['start']
    for t, th, p in zip(res['tilt_history'], res['thickness_history'], res['power_history']):
        trajectories.append({
            'Initial Tilt Angle': start_tilt,
            'Initial Thickness': start_thick,
            'Tilt Angle': t,
            'Thickness': th,
            'Power (kW)': p
        })
filtered_trajectories = pd.DataFrame(trajectories)

# Contour plot data
pivot_table = pd.read_csv(f"{output_dir}/loop_results_with_gradients.csv").pivot(
    index='Tilt Angle (degrees)', columns='Thickness (nm)', values='Power (kWh/m²/a)'
)
X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
Z = pivot_table.values

fig, ax = plt.subplots(figsize=(18, 12))
contour = ax.contourf(X, Y, Z, cmap='viridis', alpha=0.5, levels=100)
cbar = plt.colorbar(contour, ax=ax)
cbar.set_label('Energy Yield (kWh/m²/a)', fontsize=14)
cbar.ax.tick_params(labelsize=14)

# Plot trajectories
initial_conditions = filtered_trajectories[['Initial Tilt Angle', 'Initial Thickness']].drop_duplicates()
cmap = plt.get_cmap('coolwarm', len(initial_conditions))
for i, (tilt, thickness) in enumerate(initial_conditions.itertuples(index=False)):
    data = filtered_trajectories[
        (filtered_trajectories['Initial Tilt Angle'] == tilt) &
        (filtered_trajectories['Initial Thickness'] == thickness)
    ]
    tilt_angles = data['Tilt Angle'].values
    thickness_values = data['Thickness'].values
    powers = data['Power (kW)'].values
    full_tilt = np.insert(tilt_angles, 0, tilt)
    full_thickness = np.insert(thickness_values, 0, thickness)
    ax.scatter(thickness, tilt, color=cmap(i), marker='s', s=150, label=f'Initial: {powers[0]:.2f}', edgecolor='black')
    ax.scatter(thickness_values[-1], tilt_angles[-1], color=cmap(i), marker='o', s=150, label=f'Final: {powers[-1]:.2f}', edgecolor='black')
    ax.scatter(thickness_values, tilt_angles, color=cmap(i), marker='x', alpha=0.7)
    ax.plot(full_thickness, full_tilt, color=cmap(i), linestyle='-', alpha=0.7)

ax.set_xlabel('Thickness (nm)', fontsize=14)
ax.set_ylabel('Tilt Angle (degrees)', fontsize=14)
ax.legend(bbox_to_anchor=(1, 1), fontsize=14)
ax.grid()
plot_path = f"{output_dir}/{opt_method}_optimization_results_with_trajectory.pdf"
fig.tight_layout()
fig.savefig(plot_path)
plt.show()
logging.info(f"Trajectory plot saved to {plot_path}")
