from functools import partial
import jax.numpy as jnp
import jax

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import mu_0, c as c0, hbar

import os
import meow
import meow.eme.propagate
import sys
# Add src to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from SolDi2T.optics.optics_utils import *
sax_backend = meow.eme.sax._validate_sax_backend("klu")


LOG=True
material_folder = "data/optics"

df_eps_imag = pd.read_table(f"{material_folder}/reduced_eps_imag_average_mixture_1Y6_2p85PM6.txt", sep=" ", header=None)
df_eps_imag.columns = ['wl', 'eps']

df_eps_real = pd.read_table(f"{material_folder}/reduced_eps_real_average_mixture_1Y6_2p85PM6.txt", sep=" ", header=None)
df_eps_real.columns = ['wl', 'eps']
wl_active_material = jnp.array(df_eps_imag['wl'].to_numpy()) # in um

wl_ref = wl_active_material/1000 # in nm

# print(f"{wl_ref=}")

# %%
eps_pedot_reference = get_material(material_folder, "pedot_pss", wl_ref)
eps_zno_reference = get_material(material_folder, "ZnO", wl_ref)
eps_ito_reference = get_material(material_folder, "ITO", wl_ref)
eps_ag_reference = get_material(material_folder, "Ag", wl_ref)

# glass
eps_glass=1.5**2
eps_glass_reference = jnp.ones(len(wl_ref)) * eps_glass

# air
eps_air = 1.0
eps_air_reference = jnp.ones(len(wl_ref)) * eps_air

eps_pm6y6_reference = jnp.array((df_eps_real['eps'] + 1j*df_eps_imag['eps']).to_numpy())


plt.figure()# figsize=(10,3))
active_material_thickness = 0.3
ds = jnp.array([1, 1, 0.15, 0.03, active_material_thickness, 0.05, 0.01, 0.14, 1])
# with ag; the block is: glass, ito, zno, pm6y6 (original = 0.2), pedot, ag, pedot
active_material_idx = 4

#colors = mpl.cm.viridis(np.linspace(0,1,len(wl_ref)))
thetas = jnp.arange(0, 90, 1)

eps_array = jnp.array([
    eps_glass_reference, 
    eps_ito_reference, 
    eps_zno_reference, 
    eps_pm6y6_reference, 
    eps_pedot_reference, 
    eps_ag_reference
])


res_s = run_optics('s', eps_array, ds, thetas, wl_ref)
# Unpack the results
x_all_s, field_all_s, absorption_all_s, T_all_s, R_all_s = res_s

res_p = run_optics('p', eps_array, ds, thetas, wl_ref)
# Unpack the results
x_all_p, field_all_p, absorption_all_p, T_all_p, R_all_p = res_p

interpolated_absorptance_s = optics_processing(x_all_s, field_all_s, absorption_all_s, T_all_s, R_all_s, wl_ref, thetas, ds, active_material_idx, active_material_thickness)

interpolated_absorptance_p = optics_processing(x_all_p, field_all_p, absorption_all_p, T_all_p, R_all_p, wl_ref, thetas, ds, active_material_idx, active_material_thickness)

final_absorptance = (interpolated_absorptance_p + interpolated_absorptance_s)/2
print(final_absorptance.shape)
A = jnp.array(final_absorptance).reshape(1,90,901)
print(A.shape)


# Convert to Pandas DataFrame
A_df = pd.DataFrame(final_absorptance)

# Construct filename
thickness_nm = int(active_material_thickness * 1000)
output_dir = "data/absorption"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, f"A_{thickness_nm}.csv")

# Save to CSV
A_df.to_csv(filename, index=False, header=False)

print(f"✅ Saved absorptance to {filename}")
