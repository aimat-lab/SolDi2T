import sys
import os
import numpy as np
import jax.numpy as jnp
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
import time

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
# User-defined constants
# ==============================
lambda_ = jnp.arange(300, 1201, 1)
location = "seattle"
rotation_angle = 180.0
CE_value = 0.84
NOCT_value = 48.0

thickness_range = np.arange(100.0, 301.0, 1.0)  # 100–300 nm, step 20.0
tilt_angles = np.arange(0.0, 91.0, 1.0)          # 0–90°, step 1.0

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

# ==============================
# Load models
# ==============================
# Absorption surface model
loaded_absorption_surface_model, loaded_absorption_surface_params, loaded_absorption_surface_scaler_X = load_absorption_surface_model(
    "data/absorption/absorption_surface_model.pkl"
)

# Drift–diffusion electrical model
model = MultiOutputNN()
loaded_params, loaded_scaler_X, loaded_scaler_voc, loaded_scaler_jsc, loaded_scaler_ff = load_model(
    "data/electrics/drift_diffusion_nn_model.pkl"
)

# ==============================
# Prepare output directory
# ==============================
output_dir = os.path.join("data", "results", location)
os.makedirs(output_dir, exist_ok=True)

filename = "loop_results.csv"

results = []  # (Tilt Angle, Thickness, Power in kW)

# ==============================
# Begin parameter sweep
# ==============================
start_time = time.time()
logging.basicConfig(level=logging.INFO)
logging.info(f"Starting simulation loop for {location} ...")

for tilt_angle in tilt_angles:
    for target_thickness in thickness_range:
        # Predict absorption for this thickness
        A = predict_absorption_surface_NN(
            target_thickness,
            loaded_absorption_surface_model,
            loaded_absorption_surface_params,
            loaded_absorption_surface_scaler_X,
        )

        # Trim irradiance to match wavelength grid
        IdifN = trim_irradiance(lambda_, IrradianceDifN, lambda_values)
        IdirN = trim_irradiance(lambda_, IrradianceDirN, lambda_values)

        # Initialize irradiance array
        S = jnp.zeros(8760)
        NOCT = NOCT_value * jnp.ones(A.shape[0])
        CE = CE_value * jnp.ones(A.shape[0])

        # Compute Jsc
        Jsc_direct, Jsc_diffuse, Jsc, S = JscCalc_jax(
            thetasun, phisun, IdirN, IdifN, A, tilt_angle, rotation_angle, CE, S
        )

        # Compute module temperature
        temp_module = jnp.zeros((8760, A.shape[0]))
        for k in range(A.shape[0]):
            temp_module = temp_module.at[:, k].set(TempAmbient + (NOCT[k] - 20) / 800 * S)

        # Prepare drift-diffusion model input
        suns_values = S / 1000
        T_values = temp_module + 273.15

        instances = jnp.column_stack(
            (
                T_values,
                suns_values,
                jnp.full_like(suns_values, target_thickness),
            )
        )

        # Model prediction
        instances_scaled = loaded_scaler_X.transform(instances)
        pred_scaled = model.apply(loaded_params, jnp.array(instances_scaled))

        # Inverse-transform predictions
        voc_preds = loaded_scaler_voc.inverse_transform(pred_scaled[:, [0]])
        ff_preds = loaded_scaler_ff.inverse_transform(pred_scaled[:, [2]])

        # Flatten results
        voc_values = voc_preds.squeeze()
        ff_values = ff_preds.squeeze()
        jsc_values = Jsc.reshape(-1)

        # Power and yield
        power = 10 * voc_values * ff_values * jsc_values
        power_total = jnp.sum(power) / 1000  # kWh/a/m²

        results.append([tilt_angle, target_thickness, float(power_total)])

        logging.info(f"Tilt {tilt_angle:.1f}°, Thickness {target_thickness:.1f} nm → Power {power_total:.3f} kWh/m²/a")

# ==============================
# Save results and visualize
# ==============================
results_df = pd.DataFrame(results, columns=["Tilt Angle (degrees)", "Thickness (nm)", "Power (kWh/m²/a)"])
results_path = os.path.join(output_dir, filename)
results_df.to_csv(results_path, index=False)

logging.info(f"✅ Saved results to {results_path}")

# Create contour plot
pivot_table = results_df.pivot(index="Tilt Angle (degrees)", columns="Thickness (nm)", values="Power (kWh/m²/a)")
X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
Z = pivot_table.values

fig = plt.figure(figsize=(10, 6))
cp = plt.contourf(X, Y, Z, cmap="viridis")
plt.colorbar(cp, label="Power (kWh/m²/a)")
plt.xlabel("Thickness (nm)")
plt.ylabel("Tilt Angle (degrees)")
plt.title(f"Power Generation as a Function of Tilt Angle and Thickness ({location.title()})")
plot_path = os.path.join(output_dir, "contour_plot.png")
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.show()

end_time = time.time()
logging.info(f"Total runtime: {end_time - start_time:.2f} seconds")
