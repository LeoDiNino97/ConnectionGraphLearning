"""
data_utils.py

Module for "Learning the structure of connection graphs", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np
from scipy.spatial import cKDTree

import gstools as gs
import pyvista as pv

from sklearn.linear_model import OrthogonalMatchingPursuit as OMP

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.linalg import svd

import scipy.sparse as sp

from scipy.linalg import expm
from scipy.integrate import quad


###########################################
########### COMPRESSION METHODS ###########
###########################################

def vecOMP(Y, D, T0):
    
    batch_size, _ = Y.shape
    dictionary_dim = D.shape[1]

    # Initialize the coefficient matrix
    X = np.zeros((batch_size, dictionary_dim), dtype=np.float64)

    # Initialize the OMP model
    omp = OMP(n_nonzero_coefs=T0)

    # Loop through each sample in the batch
    for i in range(batch_size):
        # Fit the model to each sample
        omp.fit(D, Y[i,:])

        # Get the estimated coefficients for the current sample
        X[i, :] = omp.coef_
    
    return np.linalg.norm(Y.T - D @ X.T) ** 2 / np.linalg.norm(Y) ** 2

def LinearApproximation(U, X, M, X_true = None):
    # Compute the transform (coefficients in the U basis)
    alpha = U.T @ X
    
    # Compute the reconstruction using the first M components
    X_reconstructed = U[:, :M] @ alpha[:M, :]
    
    # Compute the reconstruction error
    if X_true is None:
        err = np.linalg.norm(X - X_reconstructed)**2 / np.linalg.norm(X)**2
    else:
        err = np.linalg.norm(X_true - X_reconstructed)**2 / np.linalg.norm(X_true)**2  

    return err

def NonlinearApproximation(U, X, M, X_true = None, return_db = False):
    # Compute the transform coefficients
    alpha = U.T @ X
    
    # Flatten and sort indices by magnitude (per column)
    idx_sorted = np.argsort(-np.abs(alpha), axis=0)  # Descending order
    
    # Create a mask to keep only M largest elements per column
    mask = np.zeros_like(alpha, dtype=bool)
    for i in range(alpha.shape[1]):  
        mask[idx_sorted[:M, i], i] = True  
    
    # Zero out non-selected elements
    alpha_thresh = alpha * mask
    
    # Compute reconstruction using the thresholded coefficients
    X_reconstructed = U @ alpha_thresh
    
    # Compute the reconstruction error
    if X_true is None:
        err = np.linalg.norm(X - X_reconstructed)**2 / np.linalg.norm(X)**2
    else:
        err = np.linalg.norm(X_true - X_reconstructed)**2 / np.linalg.norm(X_true)**2

    if return_db:
        err_db = 10 * np.log10(err)
        return err_db
    else:
        return err


# Some minor utils
def add_noise_snr(x, snr_db):
    # Signal power
    P_signal = np.mean(np.abs(x)**2)
    # Required noise power
    P_noise = P_signal / (10**(snr_db / 10))
    # Generate noise
    noise = np.sqrt(P_noise) * np.random.randn(*x.shape)
    return x + noise

def dct_basis(N):
    n = np.arange(N)
    k = n.reshape((N, 1))
    alpha = np.sqrt(2 / N) * np.ones(N)
    alpha[0] = np.sqrt(1 / N)
    C = alpha[:, None] * np.cos(np.pi * k * (n + 0.5) / N)
    return C

def random_vector_fields(points, Sigma, M, seed=None):
    # Create RNG
    rng = np.random.default_rng(seed)

    # Build mesh
    mesh = pv.PolyData(points)

    # Model for spatial correlation
    model = gs.Gaussian(dim=3, var=1.0, len_scale=0.5)
    srf = gs.SRF(model, mean=(0, 0, 0), generator="VectorField")

    # Cholesky
    L = np.linalg.cholesky(Sigma)
    vec_fields = np.empty((M, points.shape[0], 3))

    for m in range(M):
        # Use the same RNG for SRF
        f = srf.mesh(mesh, points="points", seed=rng.integers(1e9))
        vec_fields[m] = (L @ f).T

    vec_fields = vec_fields.reshape(M, 3 * points.shape[0])
    return vec_fields.T


def low_pass_filter(L, k):
    # Eigen-decomposition
    Lambda, _ = np.linalg.eigh(- L)

    # Desired frequency response (O(lambda^(-2)))
    FR = np.ones_like(Lambda)
    nonzero = (Lambda > 1e-12)

    FR[nonzero] = 1 / (np.abs(Lambda[nonzero]) ** 0.2)

    # Vandermonde matrix with increasing powers
    V = np.vander(Lambda, k, increasing=True)

    # Solve for the coefficients in the polynomial
    h, *_ = np.linalg.lstsq(V, FR, rcond=None)

    # Build polynomial filter
    PS = sum(h[p] * np.linalg.matrix_power(L, p) for p in range(k))

    return PS

def lowspace_projection(L, F, X):
    # Eigen-decomposition
    _, M = np.linalg.eigh(- L )

    # Build the projector
    P = M[:, 0 : F]
    Q = P @ P.T

    # Project the signal on the subspace spanned by the chosen frequency band 
    return Q @ X 

def xi_time_averaged(L, L_hat, T_max=50.0, num_points=1000):
    """
    Compute a stable approximation of the time-averaged distance
    xi(L, L_hat) = lim_{T->∞} (1/T) ∫_0^T ||e^{-t L} - e^{-t L_hat}||_F^2 dt
    
    Parameters
    ----------
    L : np.ndarray
        Laplacian matrix (n x n)
    L_hat : np.ndarray
        Laplacian matrix (n x n)
    T_max : float
        Maximum integration time to approximate the limit
    num_points : int
        Number of time points for numerical integration

    Returns
    -------
    xi_val : float
        Approximated value of xi(L, L_hat)
    """
    t_values = np.linspace(0, T_max, num_points)
    dt = t_values[1] - t_values[0]

    integrand = np.zeros_like(t_values)
    for i, t in enumerate(t_values):
        diff = expm(-t * L) - expm(-t * L_hat)
        integrand[i] = np.linalg.norm(diff, 'fro')**2

    xi_val = np.sum(integrand) * dt / T_max
    return xi_val

def minAICdetector(
        C, 
        M, 
        d=1, 
        eps=1e-12
        ):

    eigvals = np.sort(np.linalg.eigvalsh(C))[::-1]
    eigvals = np.maximum(eigvals, eps)
    V = len(eigvals)

    Kmax = V // d

    AICs = np.zeros(Kmax)

    for k in range(Kmax):
        rank_k = k * d

        noise_eigs = eigvals[rank_k:]
        p_k = len(noise_eigs)

        log_geo = np.mean(np.log(noise_eigs))
        log_arith = np.log(np.mean(noise_eigs))
        ratio_log = log_geo - log_arith

        AICs[k] = -2 * M * p_k * ratio_log + 2 * rank_k * (2 * V - rank_k) + 1

    k_AIC = np.argmin(AICs)
    
    # We emit the kernel dimension
    return Kmax - k_AIC