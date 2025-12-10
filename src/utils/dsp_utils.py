""" data_utils.py

Module for "Learning the structure of connection graphs", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
from scipy.linalg import expm


###########################################
########### COMPRESSION METHODS ###########
###########################################


def vec_OMP(
    Y : np.ndarray, 
    D : np.ndarray, 
    T0 : int
    ) -> float:
    """ Vectorized Orthogonal Matching Pursuit for non linear compression

    Parameters 
    ----------
    Y : np.ndarray
        Signals to be processed in the shape (num_samples, signal_dimension)
    D : np.ndarray
        Representation dictionary
    T0 : int
        Sparsity Level

    Returns
    -------
    float
        Relative reconstruction error 
    """
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


def linear_approximation(
    U : int, 
    X : int, 
    M : int, 
    X_true : np.ndarray = None
    ) -> float:
    """ Linear compression method

    Parameters 
    ----------
    U : np.ndarray
        Representation dictionary
    X : np.ndarray
        Signals to be processed in the shape (num_samples, signal_dimension)
    M : int
        Sparsity Level
    X_true : np.ndarray
        Ground-truth signals

    Returns
    -------
    float
        Relative reconstruction error 
    """
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


def nonlinear_approximation(
    U : np.ndarray, 
    X : np.ndarray, 
    M : int, 
    X_true : np.ndarray = None, 
    return_db  : bool = False
    ) -> np.ndarray:
    """ Nonlinear compression method (top eigen K)

    Parameters 
    ----------
    U : np.ndarray
        Representation dictionary
    X : np.ndarray
        Signals to be processed in the shape (num_samples, signal_dimension)
    M : int
        Sparsity Level
    X_true : np.ndarray
        Ground-truth signals
    return_db : bool
        Flag to return the error in decibel

    Returns
    -------
    float
        Relative reconstruction error 
    """
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

###################################
########### DSP METHODS ###########
###################################


def dct_basis(
        N : int
    ) -> np.ndarray:
    """ Build an N dimensional Discrete Cosine Transform basis

    Parameters
    ----------
    N : int
        Basis dimension
    
    Returns
    -------
    np.ndarray
        DCT basis
    """

    n = np.arange(N)
    k = n.reshape((N, 1))
    alpha = np.sqrt(2 / N) * np.ones(N)
    alpha[0] = np.sqrt(1 / N)
    C = alpha[:, None] * np.cos(np.pi * k * (n + 0.5) / N)
    return C

def add_noise_snr(
    x : np.ndarray, 
    snr_db : float
    ) -> np.ndarray:
    """ Adds noise to a signal accordingly to a given SNR

    Parameters
    ----------
    x : np.ndarray
        Signal to be corrupted with AWGN
    snr_db : float
        SNR level

    Returns
    -------
    np.ndarray
        Corrupted signal
    """
    # Signal power
    P_signal = np.mean(np.abs(x)**2)

    # Required noise power
    P_noise = P_signal / (10**(snr_db / 10))

    # Generate noise
    noise = np.sqrt(P_noise) * np.random.randn(*x.shape)

    return x + noise


def low_pass_filter(
    L : np.ndarray, 
    k : int
    ) -> np.ndarray:
    """ Implements a low pass filter as a FIR filter in a given Connection Laplacian with a O(k**(-2)) spectral response

    Parameters
    ----------
    L : np.ndarray
        Given Connection Laplacian
    k : int 
        Polynomial degree for FIR filter

    Returns
    -------
    np.ndarray 
        Estimated FIR filter
    """

    # Eigen-decomposition
    Lambda, _ = np.linalg.eigh(L)

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


def lowspace_projection(
    L : np.ndarray, 
    F : int, 
    X : np.ndarray
    ) -> np.ndarray:
    """ Projects a signal into the linear supsace spanned by the first F eigenvectors of a connection Laplacian

    Parameters
    ----------

    L : np.ndarray
        Given connection Laplacian
    F : int
        Cutoff frequency label
    X : np.ndarray
        Signals to be projected (nV, num_samples)

    Returns
    -------
    np.ndarray
        Projected signals
    """

    # Eigen-decomposition
    _, M = np.linalg.eigh(L )

    # Build the projector
    P = M[:, 0 : F]
    Q = P @ P.T

    # Project the signal on the subspace spanned by the chosen frequency band 
    return Q @ X 


def minAICdetector(
    C : np.ndarray, 
    M : int, 
    d : int = 1, 
    eps : float =1e-12
    ) -> int:
    """ Implements a robust detector for the kernel of a covariance matrix as a minimization of the Akaike Information Criterion

    Parameters
    ----------
    C : np.ndarray
        Covariance matrix
    M : int
        Number of observed signals
    d : int
        Local observation dimension
    eps : float
        Stability corrector
    
    Returns
    -------
    int
        Estimated kernel dimension
    """
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


def xi_time_averaged(
    L : np.ndarray, 
    L_hat : np.ndarray, 
    T_max : float = 50.0, 
    num_points : int = 1000
    ) -> float:
    """ Compute a stable approximation of the time-averaged diffusion distance
    
    Parameters
    ----------
    L : np.ndarray
        Ground truth Laplacian matrix (n x n)
    L_hat : np.ndarray
        Estimated Laplacian matrix (n x n)
    T_max : float
        Maximum integration time to approximate the limit
    num_points : int
        Number of time points for numerical integration

    Returns
    -------
    float
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

