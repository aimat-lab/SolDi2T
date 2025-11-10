import jax
import jax.numpy as jnp
from jax import jit, vmap
import jax.random as jrandom
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from SolDi2T.energy_yield.irradiance import *

# Initialize parameters
lambda_ = jnp.arange(300, 1201, 1)

q, h, c = 1.6021766208e-19, 6.62607004e-34, 299792458
const = q / (h * c) * 1e-10

@jit
def compute_hour(carry, j):
    EY_direct, EY_diffuse, EY_total, IdirN, IdifN, thetasun, phisun, A, CE, GI, GI_inv, theta, dtheta, dphi, dlambda = carry

    # Compute indices safely using JAX operations
    idx_phisun = jnp.mod(jnp.round(phisun[j]), 360).astype(jnp.int32)  # Scalar
    idx_thetasun = jnp.clip(jnp.round(thetasun[j]).astype(jnp.int32), 0, 89)  # Scalar

    # Generalized computation for k = 0 and k = 1
    def compute_k(k):
        cond_direct = (idx_thetasun <= 90) & (GI[idx_thetasun, idx_phisun] == 1)
        cond_back_diffuse = (90 - idx_thetasun < 90) & (thetasun[j] <= 90)
        index_back = jnp.clip(90 - idx_thetasun, 0, 89)

        # Direct contribution
        direct = jax.lax.cond(
            cond_direct,
            lambda _: (A[k, idx_thetasun] * IdirN[j, :] * (CE[k] * lambda_)).sum() * jnp.cos(jnp.deg2rad(thetasun[j])) * dlambda * const,
            lambda _: jax.lax.cond(
                cond_back_diffuse,
                lambda _: (A[k, index_back] * IdirN[j, :] * (CE[k] * lambda_)).sum() * jnp.cos(jnp.deg2rad(90 - thetasun[j])) * dlambda * const,
                lambda _: 0.0,
                operand=0.0  # Valid operand
            ),
            operand=0.0  # Valid operand
        )
        

        # Diffuse contribution
        tmp_diffuse = jnp.dot(
            (IdifN[j, :] * lambda_).reshape(-1, 1),
            (jnp.sum(GI[:90, :], axis=1) * jnp.sin(theta) * jnp.cos(theta)).reshape(1, -1)
        )  # Shape: (181, 90)
        diffuse = (CE[k] * (A[k, 0, :].reshape(-1, 1) * tmp_diffuse)).sum()# * dphi * dtheta * dlambda  # Scalar

        # Back Diffuse contribution
        #tmp_diffuse_back = jnp.dot(
        #    (IdifN[j, :] * lambda_).reshape(-1, 1),
        #    (jnp.sum(GI_inv[:90, :], axis=1) * jnp.sin(theta) * jnp.cos(theta)).reshape(1, -1)
        #)  # Shape: (181, 90)
        #diffuse_back = (CE[k] * (A[k, 1, :].reshape(-1, 1) * tmp_diffuse_back)).sum() * dphi * dtheta * dlambda  # Scalar
        diffuse_total = diffuse
        #diffuse_total = diffuse*0.9931485169589761-33429.43716008961# + diffuse_back  # Scalar

        diffuse_total = diffuse_total * dphi * dtheta * dlambda * const

        return direct, diffuse_total
    
    direct_0, diffuse_0 = compute_k(0)
    EY_direct_updated = EY_direct.at[j, 0].set(direct_0)
    EY_diffuse_updated = EY_diffuse.at[j, 0].set(diffuse_0)
    EY_total_updated = EY_total.at[j, 0].set(direct_0 + diffuse_0)
    
    for k in range(1,len(A)):
        direct_k, diffuse_k = compute_k(k)
        EY_direct_updated = EY_direct_updated.at[j, k].set(direct_k)
        EY_diffuse_updated = EY_diffuse_updated.at[j, k].set(diffuse_k)
        EY_total_updated = EY_total_updated.at[j, k].set(direct_k + diffuse_k)

    # Return the updated carry and a placeholder (None)
    return (
        EY_direct_updated,
        EY_diffuse_updated,
        EY_total_updated,
        IdirN,
        IdifN,
        thetasun,
        phisun,
        A,
        CE,
        GI,
        GI_inv,
        theta,
        dtheta,
        dphi,
        dlambda
    ), None

@jit
def JscCalc_jax(thetasun0_0, phisun0_0, IdirN, IdifN, A, tilt_angle, rotation_angle, CE, S):
    

    # Constants
    theta = jnp.deg2rad(jnp.linspace(0, 89, 90))  # Shape: (90,)
    dtheta = jnp.deg2rad(1)  # Scalar
    dphi = jnp.deg2rad(1)  # Scalar
    dlambda = lambda_[1] - lambda_[0]  # Scalar, typically 5 nm


    GI = get_illumination(rotation_angle, tilt_angle, 0)
    #GI = pd.read_csv('GI.csv', header=None).values
    GI_inv = 1 - GI

    # Circular shift and averaging
    #thetasun0_shifted = jnp.roll(thetasun0_0, 1)  # Circular shift by 1 position
    #phisun0_shifted = jnp.roll(phisun0_0, 1)  # Circular shift by 1 position

    # Compute the average of current and shifted arrays
    #thetasun0 = (thetasun0_0 + thetasun0_shifted) / 2
    #phisun0 = (phisun0_0 + phisun0_shifted) / 2

    #phisun, thetasun = rotatesunangle(rotation_angle, tilt_angle, 0, phisun0, thetasun0)
    phisun, thetasun = rotatesunangle(rotation_angle, tilt_angle, 0, phisun0_0, thetasun0_0)

    hemisphere_integral = jnp.sum(jnp.ones((90, 361)), axis=1) * jnp.sin(theta) * jnp.cos(theta)
    #hemisphere_integral = jnp.sum(GI[:90, :], axis=1) * jnp.sin(theta) * jnp.cos(theta)
    # Convert thetasun to radians once for vectorized cosine calculation
    cos_thetasun = jnp.cos(jnp.deg2rad(thetasun))

    # Direct contribution: broadcasting to match dimensions
    direct_contrib = IdirN * cos_thetasun[:, None]  # (8760, 181)

    # Diffuse contribution: calculate the same for each hour
    diffuse_contrib = IdifN * jnp.sum(hemisphere_integral) * dphi * dtheta  # (8760, 181)

    # Total incident solar radiation for each hour
    S = jnp.sum(direct_contrib + diffuse_contrib, axis=1) * dlambda  # (8760,)


    # Initialize the carry tuple with separate EY components
    carry = (
        jnp.zeros((8760, len(A))),  # JscDirect: (8760, 2)
        jnp.zeros((8760, len(A))),  # JscDiffuse: (8760, 2)
        jnp.zeros((8760, len(A))),  # Jsc: (8760, 2)
        IdirN,  # Shape: (8760, 181)
        IdifN,  # Shape: (8760, 181)
        thetasun,  # Shape: (8760,)
        phisun,  # Shape: (8760,)
        A,  # Shape: (2, 90, 181)
        CE,  # Shape: (2,)
        GI,  # Shape: (90, 360)
        GI_inv,  # Shape: (90, 360)
        theta,  # Shape: (90,)
        dtheta,  # Scalar
        dphi,  # Scalar
        dlambda  # Scalar
    )

    # JIT compile the compute_hour function for better performance
    compute_hour_jit = jax.jit(compute_hour)

    # Run the scan over the hours (0 to 8759)
    EY_final, _ = jax.lax.scan(compute_hour_jit, carry, jnp.arange(8760))

    # Extract the updated EY components
    Jsc_direct = EY_final[0]   # Shape: (8760, 2)
    Jsc_diffuse = EY_final[1]  # Shape: (8760, 2)
    Jsc = EY_final[2]      # Shape: (8760, 2)

    return Jsc_direct, Jsc_diffuse, Jsc, S