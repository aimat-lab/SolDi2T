import sys
import os
import numpy as np
import jax
import jax.numpy as jnp
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import time
import gc

# Disable JAX preallocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Add src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from SolDi2T.energy_yield.irradiance import trim_irradiance, get_irradiance_data
from SolDi2T.energy_yield.jsc_calculation import JscCalc_jax
from SolDi2T.optics.interpolate_absorption import (
    AbsorptionSurfaceModel,
    load_absorption_surface_model,
    predict_absorption_surface_NN,
)
from SolDi2T.electrics.model_utils import MultiOutputNN, load_model

get_irradiance_data()
# ==============================
# User parameters
# ==============================
lambda_ = jnp.arange(300, 1201, 1)
location = "phoenix"
rotation_angle = 180.0
CE_value = 0.84
NOCT_value = 48.0

thickness_range = np.arange(100.0, 310.0, 10.0)  # nm
tilt_angles = np.arange(0.0, 100.0, 10.0)         # degrees

# ==============================
# Load irradiance data
# ==============================
with open(f"data/irradiance/{location}_irradiance_data.pkl", "rb") as file:
    irradiance = pickle.load(file)

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

# Trim to model wavelength grid
IdifN = trim_irradiance(lambda_, IrradianceDifN, lambda_values)
IdirN = trim_irradiance(lambda_, IrradianceDirN, lambda_values)

# ==============================
# Load models
# ==============================
loaded_absorption_surface_model, loaded_absorption_surface_params, loaded_absorption_surface_scaler_X = load_absorption_surface_model(
    "data/absorption/absorption_surface_model.pkl"
)

model = MultiOutputNN()
loaded_params, loaded_scaler_X, loaded_scaler_voc, loaded_scaler_jsc, loaded_scaler_ff = load_model(
    "data/electrics/drift_diffusion_nn_model.pkl"
)

# ==============================
# Output directory
# ==============================
output_dir = os.path.join("data", "results", location)
os.makedirs(output_dir, exist_ok=True)

# ==============================
# Energy yield wrapper
# ==============================
def EYCalc_wrapper(tilt_angle, thickness_value):
    """Compute annual energy yield (kWh/m²/a) for given tilt & thickness."""
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

    instances_scaled = (instances - jnp.array(loaded_scaler_X.mean_)) / jnp.array(
        loaded_scaler_X.scale_
    )
    pred_scaled = model.apply(loaded_params, instances_scaled)
    voc_preds = pred_scaled[:, 0] * loaded_scaler_voc.scale_ + loaded_scaler_voc.mean_
    ff_preds = pred_scaled[:, 2] * loaded_scaler_ff.scale_ + loaded_scaler_ff.mean_

    jsc_values = Jsc.reshape(-1)
    power = 10 * voc_preds * ff_preds * jsc_values
    power_total = jnp.sum(power) / 1000  # kWh/m²/a

    return power_total

# Gradient function
grad_EYCalc = jax.grad(EYCalc_wrapper, argnums=(0, 1))

# ==============================
# Run sweep
# ==============================
start_time = time.time()
logging.basicConfig(level=logging.INFO)
logging.info(f"Starting gradient simulation loop for {location}...")

results = []

for tilt_angle in tilt_angles:
    for thickness in thickness_range:
        tilt_angle = float(tilt_angle)
        thickness = float(thickness)

        power_total = EYCalc_wrapper(tilt_angle, thickness)
        grad_tilt, grad_thickness = grad_EYCalc(tilt_angle, thickness)

        # Force computation
        power_total.block_until_ready()
        grad_tilt.block_until_ready()
        grad_thickness.block_until_ready()

        results.append([
            tilt_angle,
            thickness,
            float(power_total),
            float(grad_tilt),
            float(grad_thickness)
        ])

        logging.info(
            f"Tilt {tilt_angle:.1f}°, Thickness {thickness:.1f} nm → "
            f"Power {float(power_total):.3f} kWh/m²/a, "
            f"GradTilt {float(grad_tilt):.5f}, GradThick {float(grad_thickness):.5f}"
        )

        gc.collect()
        jax.clear_caches()

# ==============================
# Save CSV
# ==============================
results_df = pd.DataFrame(
    results,
    columns=[
        "Tilt Angle (degrees)",
        "Thickness (nm)",
        "Power (kWh/m²/a)",
        "Grad Tilt (kWh/deg)",
        "Grad Thickness (kWh/nm)",
    ],
)
csv_path = os.path.join(output_dir, "loop_results_with_gradients.csv")
results_df.to_csv(csv_path, index=False)
logging.info(f"✅ Saved results to {csv_path}")

# ==============================
# Contour plot
# ==============================
pivot_power = results_df.pivot(
    index="Tilt Angle (degrees)",
    columns="Thickness (nm)",
    values="Power (kWh/m²/a)"
)
pivot_grad_tilt = results_df.pivot(
    index="Tilt Angle (degrees)",
    columns="Thickness (nm)",
    values="Grad Tilt (kWh/deg)"
)
pivot_grad_thickness = results_df.pivot(
    index="Tilt Angle (degrees)",
    columns="Thickness (nm)",
    values="Grad Thickness (kWh/nm)"
)

X, Y = np.meshgrid(pivot_power.columns, pivot_power.index)
Z = pivot_power.values
grad_X = pivot_grad_thickness.values
grad_Y = pivot_grad_tilt.values

# Contour
fig, ax = plt.subplots(figsize=(10, 6))
contour = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
plt.colorbar(contour, label="Power (kWh/m²/a)")

# Quiver for gradients
angle_step = 1    # tilt step downsampling
thickness_step = 1 # thickness step downsampling
X_sparse = X[::angle_step, ::thickness_step]
Y_sparse = Y[::angle_step, ::thickness_step]
grad_X_sparse = grad_X[::angle_step, ::thickness_step]
grad_Y_sparse = grad_Y[::angle_step, ::thickness_step]

# Normalize gradients for consistent arrow scaling
norm = np.sqrt(grad_X_sparse**2 + grad_Y_sparse**2)
norm = np.where(norm == 0, 1, norm)
grad_X_unit = grad_X_sparse / norm
grad_Y_unit = grad_Y_sparse / norm

ax.quiver(
    X_sparse,
    Y_sparse,
    grad_X_unit,
    grad_Y_unit,
    color="white",
    scale=20,
    width=0.003,
    headwidth=3,
)
ax.set_xlabel("Thickness (nm)")
ax.set_ylabel("Tilt Angle (degrees)")
ax.set_title(f"Power & Gradient Directions — {location.title()}")

plot_path = os.path.join(output_dir, "contour_with_quiver.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()
logging.info(f"✅ Saved contour+quiver plot to {plot_path}")

end_time = time.time()
logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")

