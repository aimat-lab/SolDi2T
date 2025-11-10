from functools import partial
import jax.numpy as jnp
# import numpy as onp
from scipy.constants import epsilon_0, c as c0, hbar
import jax

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import mu_0, c as c0, hbar

LOG=True

# sax_backend = "fg"

import sax
import meow
import meow.eme.propagate
sax_backend = meow.eme.sax._validate_sax_backend("klu")

thetas = jnp.arange(0, 90, 1)


def fields(padded_ns, zi, forwards, backwards, z, wl, theta_0=0, pol="s"):
    """
    Calculates the E-fields within a stack at given positions x
    
    Arguments
    ---------

    padded_ns: np.ndarray[complex]
        refractive index per layer
    zi: np.ndarray[float]
        position of the interface
        The first interface is assumed at 0, and should not be provided
    forwards: np.ndarray[complex]
        Complex amplitudes of the forward propagating waves
        specified at the left side of the layer
    backwards: np.ndarray[complex]
        Complex amplitudes of the backward propagating waves
        specified at the left side of the layer
    z: np.ndarray[float]
        Positions at which the E-field should be evaluated
    wl: float
        Wavelength of interest
    theta_0: float
        Incidence angle in vacuum given in rad
    pol: str
        Polarization either "s" or "p"

    

    """
    zi = jnp.concatenate([jnp.array([-jnp.inf, 0]), zi, jnp.array([jnp.inf])])
    E_tot = jnp.zeros((len(z),), dtype=complex)
    Abs_tot = jnp.zeros_like(z)
    k0 = 2*jnp.pi/(wl)
    kx = k0 * jnp.sin(theta_0)
    sign = 1 if pol=="s" else -1 
    # For adding up the forward and backward propagating waves
    # Related to the sign convention of the field components
    
    for n, forward, backward, z_min, z_max in zip(padded_ns, forwards, backwards, zi, zi[1:]):
        has_contribution = jnp.any(jnp.logical_and(z > z_min, z < z_max))
        if not has_contribution:
            continue

        i_min = jnp.argmax(z >= z_min)
        i_max = jnp.argmax(z > z_max)
        
        # if i_max == 0:
        #     z_ = z[i_min:]
        # else:
        #     z_ = z[i_min:i_max]

        if jnp.isinf(z_min):
            z_local = z
        else:
            z_local = z - z_min

        kz = jnp.sqrt((k0*n)**2-kx**2) #Wavenumber normal to the interfaces
        E_local = forward*jnp.exp(1j * kz * z_local)
        E_local += sign*backward*jnp.exp(-1j * kz * z_local)
        
        # TODO make diffable here
        idxs = jnp.arange(len(E_tot))
        if i_max == 0:
            select = idxs>=i_min
        else:
            select = jnp.logical_and(idxs>=i_min, idxs<i_max)
        E_tot = jnp.where(select, E_local, E_tot)

        eps = n**2 

        omega = 2*jnp.pi*c0/(wl*1e-6)
        Abs_local = 0.5* eps.imag * epsilon_0 * jnp.abs(E_local)**2 * omega
        Abs_local /= hbar*omega #in number of photons

        Abs_tot = jnp.where(select, Abs_local, Abs_tot)
            
    return E_tot, Abs_tot

# %%
def fresnel_mirror_ij(ni=1.0, nj=1.0, theta_0=0, pol="s"):
    """Model a (fresnel) interface between two refractive indices

    Args:
        ni: refractive index of the initial medium
        nj: refractive index of the final
        theta: angle of incidence measured from normal in vacuum
        pol: "s" or "p" polarization
    """

    #print(f"{ni=}; {nj=}")
    theta_i = jnp.arcsin(jnp.sin(theta_0)/ni)
    theta_j = jnp.arcsin(jnp.sin(theta_0)/nj) #need to investigate
    cos_i = jnp.cos(theta_i)
    cos_j = jnp.cos(theta_j)

    if pol == "s":
        r_fresnel_ij = (ni*cos_i-nj*cos_j) / (ni*cos_i + nj*cos_j)  
        # i->i reflection
        t_fresnel_ij = 2*ni*cos_i / (ni*cos_i + nj*cos_j)  # i->j transmission
        t_fresnel_ji = 2*nj*cos_j / (ni*cos_i + nj*cos_j) 
        
    elif pol == "p":
        r_fresnel_ij = (nj*cos_i - ni*cos_j) / (nj*cos_i + ni*cos_j)  
        # i->i reflection
        t_fresnel_ij = 2*ni*cos_i / (nj*cos_i + ni*cos_j)  # i->j transmission
        t_fresnel_ji = 2*nj*cos_j / (nj*cos_i + ni*cos_j)

    else:
        raise ValueError(f"polarization should be either 's' or 'p'")
    
    r_fresnel_ji = -r_fresnel_ij  # j -> i reflection
    
    sdict = {
        ("left", "left"): r_fresnel_ij,
        ("left", "right"): t_fresnel_ij,
        ("right", "left"): t_fresnel_ji,
        ("right", "right"): r_fresnel_ji,
    }
    return sdict

def propagation_i(ni=1.0, di=0.5, wl=0.532, theta_0=0):
    """Model the phase shift acquired as a wave propagates through medium A

    Args:
        ni: refractive index of medium (at wavelength wl)
        di: [μm] thickness of layer
        wl: [μm] wavelength
        theta: angle of incidence measured from normal in vacuum
    """
    k0 = 2*jnp.pi/(wl)
    kx = k0 * jnp.sin(theta_0)
    kz = jnp.sqrt((k0*ni)**2-kx**2)

    prop_i = jnp.exp(1j * kz * di)
    sdict = {
        ("left", "right"): prop_i,
        ("right", "left"): prop_i,
    }
    return sdict

def split_square_matrix(matrix, idx):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix has to be square")
    return [matrix[:idx, :idx], matrix[:idx, idx:]], [
        matrix[idx:, :idx],
        matrix[idx:, idx:],
    ]

def propagate(l2rs, r2ls, excitation_l, excitation_r):
    forwards = []
    backwards = []
    for l2r, r2l in zip(l2rs, r2ls):
        s_l2r, p = sax.sdense(l2r)
        s_r2l, _ = sax.sdense(r2l)
        m = len([k for k in p.keys() if "right" in k])
        f, b = compute_mode_amplitudes(s_l2r, s_r2l, m, excitation_l, excitation_r)
        forwards.append(f)
        backwards.append(b)
    return forwards, backwards

def compute_mode_amplitudes(u, v, m, excitation_l, excitation_r):
    # print("#"*30)
    # print(f"{u=}")
    # print(f"{v=}")
    n = u.shape[0] - m
    l = v.shape[0] - m
    [u11, u21], [u12, u22] = split_square_matrix(u, n) 
    [v11, v21], [v12, v22] = split_square_matrix(v, m)
    #Sax uses notation where 1->2 is in entry 12.
    #The sandwich equations are derived for 1->2 in entry 21 (2 from 1)

    RHS = u21 @ excitation_l + u22 @ v12 @ excitation_r
    LHS = jnp.diag(jnp.ones(m)) - u22 @ v11
    forward = jnp.linalg.solve(LHS, RHS)
    backward = v12 @ excitation_r + v11 @ forward

    return forward, backward

# @partial(jax.jit, static_argnames="polarization")


def fields_in_stack(ds, ns, wl, theta_0, res=2000, pol="s"):
    ex_l = jnp.array([1])
    ex_r = jnp.array([0])

    identity = fresnel_mirror_ij(1, 1)

    propagations=[propagation_i(ni, di, wl, theta_0=theta_0) for ni, di in zip(ns, ds)]
    propagations=[identity]+propagations+[identity]
    propagations = {f"p_{i}": sax.sdense(p) for i, p in enumerate(propagations)}

    one = jnp.array([1])
    padded_ns = jnp.concatenate([one, ns, one])
    interfaces=[fresnel_mirror_ij(ni, nj, theta_0=theta_0, pol=pol) for ni, nj in zip(padded_ns, padded_ns[1:])]
    # print(jnp.shape(interfaces), jnp.shape(propagations))
    
    interfaces = {f"i_{i}_{i+1}": sax.sdense(p) for i, p in enumerate(interfaces)}


    pairs = meow.eme.propagate.pi_pairs(propagations, interfaces, sax_backend)
    l2rs = meow.eme.propagate.l2r_matrices(pairs, identity, sax_backend)
    r2ls = meow.eme.propagate.r2l_matrices(pairs, sax_backend)
    forwards, backwards = propagate(l2rs, r2ls, ex_l, ex_r)

    #TODO calculate absorption from bwd and fwd coefficients
    # print(f"{interfaces=}")

    # print("#"*10)
    # print(f"{l2rs=}")
    # print(f"{jnp.abs(jnp.array(forwards))}")
    # print(f"{jnp.abs(jnp.array(backwards))}")

    ds = jnp.array(ds)
    zi = jnp.cumsum(ds)
    z = jnp.linspace(0, jnp.round(jnp.sum(ds), 2), res)
    field, absorption = fields(padded_ns, zi, forwards, backwards, z, wl, 
                               theta_0=theta_0, pol=pol)
    
    T = jnp.abs(forwards[-1][0])**2
    R = jnp.abs(backwards[0][0])**2
    return z, field, absorption, T, R


def get_material(material_folder, name, wl_reference):
    data_real = pd.read_csv(f"{material_folder}/{name}_real.csv", sep=",", header=0)
    data_imag = pd.read_csv(f"{material_folder}/{name}_imag.csv", sep=",", header=0)
    wl = jnp.array(data_real["wl"].to_numpy())
    n = jnp.array(data_real["n"].to_numpy())
    #print("wl_imag shape:", wl_imag.shape)  # Should match data_imag["wl"]
    k = jnp.interp(wl, jnp.array(data_imag['wl'].to_numpy()), jnp.array(data_imag['k'].to_numpy()))
    
    eps = (n+1j*k)**2
    eps_reference = jnp.interp(wl_reference, wl, eps)
    return eps_reference


def calc_single(i_wl, theta_deg, eps_array, ds, wl_ref, pol='p'):
    """
    Calculate for a single combination of angle and wavelength
    """
    wl = wl_ref[i_wl]  # Use wl_ref from the arguments
    theta_0 = jnp.deg2rad(theta_deg)

    omega = 2 * jnp.pi * c0 / (wl * 1e-6)
    Irr = 1 / (2 * mu_0 * c0) * jnp.cos(theta_0)

    ns = jnp.concatenate([jnp.array([1]), jnp.sqrt(eps_array[:, i_wl]).reshape(-1), jnp.array([1])])

    x, field, absorption, T, R = fields_in_stack(ds, ns, wl, theta_0, pol=pol)

    return x, field, absorption, T, R


def run_optics(pol, eps_array, ds, thetas, wl_ref):
    """
    Run the optics calculation for the desired polarization and theta angles.
    """
    # Set the polarization you desire, for example "s"
    calc_single_static = partial(calc_single, pol=pol, wl_ref=wl_ref)  # Pass wl_ref here

    # Vectorize over wavelength index (i_wl) and theta (theta_deg)
    calc_single_vmapped = jax.vmap(calc_single_static, in_axes=(0, None, None, None))  # Vectorize over i_wl and theta_deg
    calc_double_vmapped = jax.vmap(calc_single_vmapped, in_axes=(None, 0, None, None))  # Vectorize over both i_wl and theta_deg

    # Convert thetas to a JAX array
    thetas_jnp = jnp.array(thetas)

    # Calculate for all wavelengths and angles at once
    res = calc_double_vmapped(jnp.arange(len(wl_ref), dtype=int), thetas_jnp, eps_array, ds)

    return res


def optics_processing(x_all, field_all, absorption_all, T_all, R_all, wl_ref, thetas, ds, active_material_idx, active_material_thickness):
    # Precompute constants

    omegas = 2 * jnp.pi * c0 / (wl_ref * 1e-6)
    theta_0s = jnp.deg2rad(thetas)
    Irrs = 1 / (2 * mu_0 * c0) * jnp.cos(theta_0s[:, None])

    # Compute start and end of active material layer
    start_active = jnp.sum(jnp.array(ds)[:active_material_idx])
    end_active = start_active + active_material_thickness

    # Compute masks for active material
    active_masks = jnp.logical_and(x_all > start_active, x_all < end_active)

    # Compute dx (assume uniform spacing)
    dx = (x_all[:, :, 1] - x_all[:, :, 0]) * 1e-6

    deltawl = wl_ref[0] - wl_ref[1]  # scalar wavelength interval
    # Sum over angles (axis 0) and wavelengths (axis 1) to get a vector of shape (2000,)
    wl_sum_absorption = jnp.sum(absorption_all * deltawl, axis=(0, 1))

    active_absorption = dx * jnp.sum(absorption_all * active_masks, axis=2)

    total_absorption = dx * jnp.sum(absorption_all, axis=2)

    # Compute final quantities
    absorption_whole_stack = total_absorption * hbar * omegas / Irrs
    absorption_active_material = active_absorption * hbar * omegas / Irrs

    # Start of processing
    wl_ref_nm = wl_ref * 1000

    # Define new wavelength range (1 nm step) for interpolation
    new_wavelengths = jnp.arange(wl_ref_nm.min(), wl_ref_nm.max() + 1, 1)

    # Interpolate along the wavelength axis for all angles at once
    interpolated_absorptance = jnp.array([
        jnp.interp(new_wavelengths, wl_ref_nm[::-1], absorption_active_material[i][::-1])
        for i in range(len(thetas))
    ])
    
    return interpolated_absorptance




