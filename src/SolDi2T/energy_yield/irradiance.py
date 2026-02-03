from __future__ import division, print_function, absolute_import
import jax.numpy as jnp
import jax
import jax.nn as jnn
from jax.scipy.spatial.transform import Rotation  # Check if this is available in your environment
import os
import requests
import zipfile
import io

degree = jnp.pi / 180

def read_irradiance(CodeLocation, AliasLocation):
    # Define the folder name of chosen location
    FolderNameIrradiance = os.path.join(os.getcwd(), f'Irradiance/Spectra_{CodeLocation}_{AliasLocation}')
    
    # Simulate the irradiance data
    if not os.path.isdir(FolderNameIrradiance):
        pass  # Placeholder for actual irradiance simulation code
    
    # Load irradiance data
    # You can replace this with the actual loading mechanism
    if 'irradiance' not in locals():
        # Placeholder for actual irradiance data loading
        irradiance = None
        IndRefr = {}
        
    return irradiance

@jax.jit
def wrap_to_180(angles):
    return (angles + 180) % 360 - 180

@jax.jit
def wrap_to_360(angles):
    return angles % 360

@jax.jit
def rotatesunangle(alpha, beta, gamma, phisun, thetasun):
    phisun = wrap_to_180(jnp.array(phisun))
    thetasun = 90 - jnp.array(thetasun)

    phisun = jnp.deg2rad(phisun)
    thetasun = jnp.deg2rad(thetasun)

    eul = jnp.deg2rad(jnp.array([alpha, beta, gamma]))
    rotm = jax.scipy.spatial.transform.Rotation.from_euler('ZYX',eul).as_quat()
    rotm = [rotm[3], rotm[0], rotm[1], rotm[2]]
    sx, sy, sz = sph2cart(phisun, thetasun, jnp.ones_like(phisun))

    s = jnp.stack((sx, sy, sz), axis=-1)
    
    rotm_scipy = [rotm[1], rotm[2], rotm[3], rotm[0]]
    coord_rot = jax.scipy.spatial.transform.Rotation.from_quat(rotm_scipy).apply(s)
    phisun_rot, thetasun_rot, _ = cart2sph(coord_rot[:, 0], coord_rot[:, 1], coord_rot[:, 2])

    thetasun_rot = jnp.rad2deg(thetasun_rot)
    thetasun_rot = 90 - thetasun_rot
    phisun_rot = jnp.rad2deg(phisun_rot)
    phisun_rot = jnp.round(wrap_to_360(phisun_rot), 5)

    return phisun_rot, thetasun_rot


def trim_irradiance(lambda_range, I, w):
    startindex = jnp.where(w == lambda_range[0])[0][0]
    stopindex = jnp.where(w == lambda_range[-1])[0][0]
    d = lambda_range[1] - lambda_range[0]

    step = int(d / (w[1] - w[0]))

    I_trimmed = I[:, startindex:stopindex + 1:step]

    return jnp.array(I_trimmed)


@jax.jit
def sph2cart(phi, theta, r):
    x = r * jnp.cos(theta) * jnp.cos(phi)
    y = r * jnp.cos(theta) * jnp.sin(phi)
    z = r * jnp.sin(theta)
    return x, y, z

@jax.jit
def cart2sph(x, y, z):
    hxy = jnp.hypot(x, y)
    r = jnp.hypot(hxy, z)
    el = jnp.arctan2(z, hxy)
    az = jnp.arctan2(y, x)
    return az, el, r

@jax.jit
def get_illumination(alpha, beta, gamma):
    # Normal of flat and unrotated solar cell pointing to zenith
    n = jnp.array([0, 0, 1])

    # Define Euler angles for rotation 'ZYX'
    eul = jnp.deg2rad(jnp.array([alpha, -beta, gamma]))
    #rotm = Rotation.from_euler('ZYX', eul).as_quat()
    rotm = jax.scipy.spatial.transform.Rotation.from_euler('ZYX',eul).as_quat()
    rotm = [rotm[3], rotm[0], rotm[1], rotm[2]]
    # Rotate normal about alpha and beta
    coord_rot = Rotation.from_quat(rotm).apply(n)

    n_new = jnp.array([coord_rot[0], coord_rot[1], coord_rot[2]])

    # Define local coordinates
    phi = jnp.linspace(-jnp.pi, jnp.pi, 361)
    theta = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, 181)

    # Make meshgrid with phi and theta
    P, T = jnp.meshgrid(phi, theta)
    R_grid = jnp.ones_like(P)

    # Transform the angle grid to Cartesian coordinates
    px, py, pz = sph2cart(P, T, R_grid)
    p = jnp.stack((px, py, pz), axis=-1)

    # Reshape the rotated normal to fit the size of the defined grid p
    n_new = jnp.tile(n_new.reshape(1, 1, 3), (p.shape[0], p.shape[1], 1))

    const1 = 10.0  # Adjust sharpness for A calculation
    const2 = 10.0  # Adjust sharpness for GI calculation

    # Sigmoid approximation for A calculation (instead of clipping)
    sigmoid_value = jnn.sigmoid(const1 * jnp.sum(n_new * p, axis=2))
    A = jnp.degrees(jnp.arccos(2.0 * sigmoid_value - 1.0))

    # GI calculation using sigmoid (instead of tanh)
    GI = jnn.sigmoid(-const2 * (A - 89))

    # Take front side of cell only
    GI = GI.at[:91, :].set(0)
    GI = jnp.flipud(GI)

    return GI

def get_irradiance_data():
    target_dir = 'data/irradiance'
    
    # 1. The official "Download All" link for the folder
    url = "https://bwsyncandshare.kit.edu/s/rrXQn4fNxGpAHXn/download"

    # Only download if the folder is missing or empty
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print(f"Downloading data from {url}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status() # Check for HTTP errors
            
            # Check if we actually got a ZIP file (and not a small error page)
            if len(response.content) < 1000: 
                print("Warning: Downloaded file is suspiciously small. Check the link.")

            print("Extracting files...")
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall('data')
                
            print("Success! Files ready.")
            
        except Exception as e:
            print(f"Error during download: {e}")
            # Clean up partial bad downloads
            if os.path.exists(target_dir):
                import shutil
                shutil.rmtree(target_dir)
    else:
        print("Files already exist. Skipping download.")