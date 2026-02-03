import jax.numpy as jnp
import pickle
import numpy as np
import pandas as pd
import sys
import os
import logging

# Add src to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from SolDi2T.energy_yield.irradiance import trim_irradiance, get_irradiance_data
from SolDi2T.energy_yield.jsc_calculation import JscCalc_jax
from SolDi2T.optics.interpolate_absorption import AbsorptionSurfaceModel, load_absorption_surface_model, predict_absorption_surface_NN
from SolDi2T.electrics.model_utils import MultiOutputNN, load_model 

import time

get_irradiance_data()

# Record the start time
start_time = time.time()

# Initialize parameters
lambda_ = jnp.arange(300, 1201, 1)  # Wavelength range: 300 nm to 1200 nm in 1 nm steps
target_thickness = 200.0 # active layer material thickness, in nm
location = 'phoenix' # location name
tilt_angle = 20.0 # tilt angle in degrees
rotation_angle = 180.0 # rotation angle in degrees
CE_value = 0.84 # collection efficiency
NOCT_value = 48.0 # nominal operating cell temperature, in °C

#loaded_absorption_model, loaded_absorption_params, loaded_absorption_scaler_X, loaded_absorption_scaler_y = load_absorption_model('data/absorption/absorption_model.pkl')
# Load it
loaded_absorption_surface_model, loaded_absorption_surface_params, loaded_absorption_surface_scaler_X = load_absorption_surface_model('data/absorption/absorption_surface_model.pkl')
# Final absorption array
A = predict_absorption_surface_NN(target_thickness, loaded_absorption_surface_model, loaded_absorption_surface_params, loaded_absorption_surface_scaler_X)


# Load irradiance data
with open('data/irradiance/'+location+'_irradiance_data.pkl', 'rb') as file:
    irradiance = pickle.load(file)

# Extract necessary irradiance data
lambda_values = irradiance['Irr_spectra_clouds_wavelength'].flatten()  # Shape: (181,)

thetasun = irradiance['Data_TMY3']['Data_TMY3'][:, 6]  # Shape: (8760,)
phisun = irradiance['Data_TMY3']['Data_TMY3'][:, 7]  # Shape: (8760,)

# Irradiance calculations
IrradianceDifH = irradiance["Irr_spectra_clouds_diffuse_horizontal"]  # Shape: (8760, 181)
IrradianceDifN = IrradianceDifH / jnp.pi

IrradianceDirH = irradiance["Irr_spectra_clouds_direct_horizontal"]  # Shape: (8760, 181)
IrradianceDirN = (jnp.ceil(thetasun) < 90)[:, jnp.newaxis] * IrradianceDirH / jnp.abs(jnp.cos(jnp.deg2rad(thetasun)))[:,
                                                                          jnp.newaxis]

# Trim irradiance to match lambda_ range
IdifN = trim_irradiance(lambda_, IrradianceDifN, lambda_values)  # Shape: (8760, 181)
IdirN = trim_irradiance(lambda_, IrradianceDirN, lambda_values)  # Shape: (8760, 181)

TempAmbient = irradiance['Data_TMY3']['Data_TMY3'][:, 13]

# Initialize irradiance array S
S = jnp.zeros(8760)

NOCT = NOCT_value * jnp.ones(A.shape[0])
CE = CE_value * jnp.ones(A.shape[0])

# Calculate Jsc from irradiance
Jsc_direct, Jsc_diffuse, Jsc, S = JscCalc_jax(thetasun, phisun, IdirN, IdifN, A, tilt_angle, rotation_angle, CE, S)

#Jsc = pd.read_csv('Jsc_300.csv', header=None).values
#S = pd.read_csv('S.csv',header=None).values

# Calculate the temperature of the module
temp_module = jnp.zeros((8760, A.shape[0]))

for k in range(A.shape[0]):
    temp_module = temp_module.at[:, k].set(TempAmbient + (NOCT[k] - 20) / 800 * S)


# Initialize the Drift-Diffusion model
model = MultiOutputNN()
# Load the model
loaded_params, loaded_scaler_X, loaded_scaler_voc, loaded_scaler_jsc, loaded_scaler_ff = load_model('data/electrics/drift_diffusion_nn_model.pkl')
# Convert irradiance values (S) from W/m^2 to suns (divide by 1000)
suns_values = S/1000

# Convert temperature from Celsius to Kelvin
T_values = temp_module + 273.15

#T_values = jnp.zeros((8760, 1))
#T_values = T_values.at[0, 0].set(298.15)

# Initialize arrays to store voc and ff values
voc_values = jnp.zeros(len(S))
ff_values = jnp.zeros(len(S))

# Prepare batch inputs
num_samples = len(suns_values)

# Prepare batch inputs
instances = jnp.column_stack((
    T_values,  # Directly use T_values
    suns_values,  # Directly use suns_values
    jnp.full_like(suns_values, target_thickness)
))

# Scale all instances at once
instances_scaled = loaded_scaler_X.transform(instances)

# Apply the model in batch
pred_scaled = model.apply(loaded_params, jnp.array(instances_scaled))

# Inverse transform predictions
voc_preds = loaded_scaler_voc.inverse_transform(pred_scaled[:, [0]])
ff_preds = loaded_scaler_ff.inverse_transform(pred_scaled[:, [2]])

# Flatten the results
voc_values = voc_preds.squeeze()
ff_values = ff_preds.squeeze()

# Reshape original Jsc array and convert it
jsc_values = Jsc.reshape(-1)

# Calculate power from voc, ff and jsc
power = (10 * voc_values * ff_values * jsc_values.squeeze())

# Calculate total energy yield
power_total = jnp.sum(power) / 1000

# Calculate global irradiance
Iglob = jnp.sum(IrradianceDifH + IrradianceDirN, axis=1)

# Calculate the power conversion efficiency (pce), which only makes sense to interpret if we are using the AM1.5 ideal spectrum
pce = power / Iglob
pce = jnp.where(jnp.isnan(pce), 0, pce)
pce = jnp.where(jnp.isinf(pce), 0, pce)
pce = jnp.where(pce < 0, 0, pce)

# Calculate the average pce
pce_mean = jnp.mean(pce)

# Output the results
logging.basicConfig(level=logging.INFO)
logging.info(f"Annual energy yield: {power_total} kWh/a/m²")
logging.info(f'The average power conversion efficiency is {pce_mean*100} %')

# Record the end time
end_time = time.time()

# Calculate the duration
total_duration = end_time - start_time
# Print the duration
logging.info(f"The full script took a total of {total_duration:.2f} seconds to run.")
