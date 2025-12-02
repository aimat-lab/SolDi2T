import jax
import jax.numpy as jnp
import pickle
import numpy as np
import sys
import os
import logging
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy.ma as ma
# # for plotting
# textwidth = 455.2 / 72 * 1.2
# import matplotlib as mpl
# # Set the font to Computer Modern Roman
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.serif'] = 'Arial' # 'Computer Modern Roman'
# # Use LaTeX to format text
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams.update({'font.size': 9.8})

# -----------------------------
# Setup
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

location = 'phoenix'
fontsize = 24
fontsize2 = 24
with_zoom = True

# Adjustable: how many steps to skip when plotting intermediate points
plot_step = 5  # Change to 1 for all steps, 10 for sparse, etc.

# -----------------------------
# Load gradient ascent results
# -----------------------------
with open(f'data/results/{location}/gradient_ascent_results.pkl', 'rb') as f:
    all_results = pickle.load(f)

# Load energy yield surface
path = f'data/results/{location}/loop_results.csv'
#path = f'data/results/{location}/loop_results_with_gradients.csv'
results_df = pd.read_csv(path)
pivot_table = results_df.pivot(index='Tilt Angle (degrees)', columns='Thickness (nm)', values='Power (kWh/m²/a)')
X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
Z = pivot_table.values  # already in kWh/m²/a

# -----------------------------
# Base contour plot
# -----------------------------
power_min, power_max = Z.min(), Z.max()
num_levels = 150
levels = np.linspace(power_min, power_max, num_levels)

fig, ax = plt.subplots(figsize=(18, 8))
contour = ax.contourf(X, Y, Z, cmap='viridis', alpha=1, levels=levels)
cbar = plt.colorbar(contour, ax=ax)
#cbar.set_label(r'Energy Yield (kWh/$m^{2}/a$)', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)

# Custom ticks only
custom_ticks = np.arange(130, 231, 20)
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([f"{t:.0f}" for t in custom_ticks])
#cbar.set_ticklabels([rf"\text{{{int(t)}}}" for t in custom_ticks])
#ax.set_xlabel('Thickness (nm)', fontsize=fontsize)
#ax.set_ylabel('Tilt Angle (degrees)', fontsize=fontsize)


# -----------------------------
# Plot gradient ascent trajectories
# -----------------------------
colors = [
    'C4', 'C1', 'C5', 'C3', 'lightsalmon', 'skyblue', 'gold',
    'mediumseagreen', 'orchid', 'coral', 'turquoise', 'plum'
]

max_power = -np.inf
best_thickness = None
best_tilt = None

for i, result in enumerate(all_results):
    tilt_h = np.array(result["tilt_history"])
    thick_h = np.array(result["thickness_history"])
    power_h = np.array(result['power_history'])

    # Plot trajectory path
    ax.plot(thick_h, tilt_h, linestyle='--', color=colors[i], linewidth=3,
            label=f'Trajectory from ({int(thick_h[0])}, {int(tilt_h[0])})')

    # Plot sampled intermediate points
    indices = list(range(0, len(thick_h), plot_step))
    if len(thick_h) - 1 not in indices:
        indices.append(len(thick_h) - 1)

    #ax.scatter(thick_h[indices], tilt_h[indices], color=colors[i], s=70, alpha=0.9, edgecolor='black', linewidth=0.5)

    # Start and end markers
    ax.scatter(thick_h[0], tilt_h[0], marker='s', s=150, color=colors[i], edgecolor='black', linewidth=1.2)
    ax.scatter(thick_h[-1], tilt_h[-1], marker='o', s=150, color=colors[i], edgecolor='black', linewidth=1.2)

    # Track best overall point
    local_max_idx = np.argmax(power_h)
    if power_h[local_max_idx] > max_power:
        max_power = power_h[local_max_idx]
        best_thickness = thick_h[local_max_idx]
        best_tilt = tilt_h[local_max_idx]

# from matplotlib.patches import Rectangle

# zoom_rect = Rectangle(
#     (160, 15),        # (x1, y1)
#     240 - 160,        # width
#     35 - 15,          # height
#     fill=False,
#     edgecolor='black',
#     linewidth=1.0,
#     linestyle='-'
# )
# ax.add_patch(zoom_rect)

ax.legend(fontsize=fontsize - 4, loc='upper right')
ax.set_xlim([100, 300])
ax.set_ylim([0, 90])
ax.tick_params(axis='both', which='major', labelsize=fontsize)

print(f"\n🌟 Maximum energy yield across all runs: {max_power:.2f} kWh/m²/a")
print(f"At thickness = {best_thickness:.2f} nm, tilt = {best_tilt:.2f}°")

# -----------------------------
# Plot gradient arrows (optional quiver overlay)
# -----------------------------
file_path = f'data/results/{location}/loop_results_with_gradients.csv'
results_df = pd.read_csv(file_path)
pivot_grad_tilt = results_df.pivot(index='Tilt Angle (degrees)', columns='Thickness (nm)', values='Grad Tilt (kWh/deg)')
pivot_grad_thickness = results_df.pivot(index='Tilt Angle (degrees)', columns='Thickness (nm)', values='Grad Thickness (kWh/nm)')

grad_X = pivot_grad_thickness.values
grad_Y = pivot_grad_tilt.values
X_grad, Y_grad = np.meshgrid(pivot_grad_thickness.columns, pivot_grad_thickness.index)

ax.quiver(X_grad, Y_grad, grad_X * 0.5, grad_Y * 0.5, color='black', scale=50, alpha=0.2, width=0.0025)

fig.tight_layout()
fig.savefig(f'data/results/{location}/gradient_ascent_full.png', dpi=300, bbox_inches='tight')
fig.savefig(f'data/results/{location}/gradient_ascent_full.pdf', dpi=300, bbox_inches='tight')
plt.show()

# -----------------------------
# Zoomed-in region plot (optional)
# -----------------------------
if with_zoom:
    x1, x2 = 160, 240
    y1, y2 = 15, 35

    fig2, ax2 = plt.subplots(figsize=(18, 6))
    idxx1 = np.argmin(np.abs(X[0] - x1))
    idxx2 = np.argmin(np.abs(X[0] - x2))
    idxy1 = np.argmin(np.abs(Y[:, 0] - y1))
    idxy2 = np.argmin(np.abs(Y[:, 0] - y2))

    Z_zoom = Z[idxy1:idxy2, idxx1:idxx2]
    vmin, vmax = Z_zoom.min(), Z_zoom.max()

    contour_zoom = ax2.imshow(
        Z_zoom, cmap='coolwarm', interpolation=None, aspect=1.5, alpha=0.7,
        extent=(x1, x2, y1, y2), origin='lower', vmin=vmin, vmax=vmax
    )

    cbar2 = plt.colorbar(contour_zoom, ax=ax2)
    #cbar2.set_label(r'Energy Yield (kWh/$m^{2}/a$)', fontsize=fontsize2)
    #cbar2.ax.tick_params(labelsize=fontsize2)
    # Use the same style as the main colorbar, but with zoomed tick range
    custom_ticks_zoom = np.arange(232, 243, 2)   # 232 → 242
    cbar2.set_ticks(custom_ticks_zoom)
    cbar2.set_ticklabels([f"{t:.0f}" for t in custom_ticks_zoom])
    cbar2.ax.tick_params(labelsize=fontsize2)

    for i, result in enumerate(all_results):
        tilt_h = np.array(result["tilt_history"])
        thick_h = np.array(result["thickness_history"])

        indices = list(range(0, len(thick_h), plot_step))
        if len(thick_h) - 1 not in indices:
            indices.append(len(thick_h) - 1)

        ax2.plot(thick_h[indices], tilt_h[indices], linestyle='--', color=colors[i], linewidth=1.5)
        #ax2.scatter(thick_h, tilt_h, color=colors[i], s=50, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax2.scatter(thick_h[0], tilt_h[0], marker='s', s=70, color=colors[i], edgecolor='black', linewidth=1.2)
        ax2.scatter(thick_h[-1], tilt_h[-1], marker='o', s=70, color=colors[i], edgecolor='black', linewidth=1.2)

    mask_zoom = (X_grad >= x1) & (X_grad <= x2) & (Y_grad >= y1) & (Y_grad <= y2)
    ax2.quiver(
        X_grad[mask_zoom], Y_grad[mask_zoom],
        grad_X[mask_zoom] * 2, grad_Y[mask_zoom] * 2,
        color='black', scale=50, alpha=0.3, width=0.0025
    )

    ax2.set_xlim(x1, x2)
    ax2.set_ylim(y1, y2)
    #ax2.set_xlabel('Thickness (nm)', fontsize=fontsize2)
    #ax2.set_ylabel('Tilt Angle (degrees)', fontsize=fontsize2)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize2)

    fig2.tight_layout()
    fig2.savefig(f'data/results/{location}/gradient_ascent_zoomed_full.png', dpi=300, bbox_inches='tight')
    fig2.savefig(f'data/results/{location}/gradient_ascent_zoomed_full.pdf', dpi=300, bbox_inches='tight')
    plt.show()
