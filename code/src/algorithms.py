import sys
sys.path.insert(1,'../')

import autograd.numpy as anp
from autograd.numpy.linalg import norm, svd, inv
from src.utils import stiefel_matrix

def proj_stiefel(M):
    """
    Projects the matrix M onto the Stiefel manifold through the polar decomposition: if M = U S V^T then return U V^T.
    
    INPUT
    =====
    M : anp.ndarray. A real matrix of shape (n_rho, n_tau).
        
    OUTPUT
    ======
    Y : anp.ndarray. The projection of M onto the Stiefel manifold (i.e. Y^T Y = I).
    """
    U, s, Vt = svd(M, full_matrices=False)
    return U @ Vt

def local_solver_stiefel(X_rho, X_sigma, n_tau, alpha, tol_abs=1.e-4, tol_rel=1.e-4, max_iter=1000, seed=42):
    """
    Solves the local (unweighted) problem by using the splitting of the orthogonality constraints (SOC) together with the alternating direction method of multipliers (ADMM).
    
    INPUT
    =====
    X_rho : anp.ndarray, shape (n_rho, d). Data matrix associated with the node rho.
    X_sigma : anp.ndarray, shape (n_sigma, d). Data matrix associated with the node rho.
    n_tau : int. Dimensionality of the stalk on the edge. In our formulation, F_rhotau is (n_tau x n_rho) and Y_rho is (n_rho x n_tau); similarly for sigma.
    alpha : float. ADMM stepsize.
    tol_abs : float. Absolute tolerance for the stopping criterion, optional (default=1.e-4).
    tol_rel : float. Relative tolerance for the stopping criterion, optional (default=1.e-4).
    max_iter : int, optional (default=1000). Maximum number of ADMM iterations.
    seed: int, optional (defalut=42). Initialization seed.
        
    OUTPUT
    ======
    F_rhotau, F_sigmatau, Y_rho, Y_sigma, U_rho, U_sigma : tuple of anp.ndarray
        The final iterates. Here:
          - F_rhotau is an array of shape (n_tau, n_rho)
          - F_sigmatau is an array of shape (n_tau, n_sigma)
          - Y_rho is an array of shape (n_rho, n_tau) 
          - Y_sigma is an array of shape (n_sigma, n_tau) 
          - U_rho and U_sigma are the scaled dual variables (same shape as Y_rho and Y_sigma, respectively).
    """
    #Dimensions
    n_rho, m_rho = X_rho.shape
    n_sigma, m_sigma = X_sigma.shape
    assert m_rho == m_sigma, "X_rho and X_sigma must have the same number of samples (columns)."
    
    #Precompute constant matrices and their inverses for updating the restriction maps
    C_rhosigma = X_rho @ X_sigma.T
    C_sigmarho = X_sigma @ X_rho.T
    R_rho = 2 * X_rho @ X_rho.T + alpha * anp.eye(n_rho)  #shape:(n_rho, n_rho)
    R_sigma = 2 * X_sigma @ X_sigma.T + alpha * anp.eye(n_sigma)  # shape:(n_sigma, n_sigma)

    R_rho_inv = inv(R_rho)    
    R_sigma_inv = inv(R_sigma)
    
    #Initialize Y_rho and Y_sigma randomly on the Stiefel manifold.
    Y_rho = stiefel_matrix(n_rho, n_tau, seed=seed)
    Y_sigma = stiefel_matrix(n_sigma, n_tau, seed=seed)
    
    # Initialize restriction maps 
    F_rhotau = Y_rho.T.copy()      
    F_sigmatau = Y_sigma.T.copy() 
    
    # Initialize dual variables U 
    U_rho = anp.zeros_like(Y_rho)
    U_sigma = anp.zeros_like(Y_sigma)
    
    # ADMM iterations
    for k in range(max_iter):
        #Store splitting values for the computation of the dual residuals
        Y_rho_prev = Y_rho.copy()
        Y_sigma_prev = Y_sigma.copy()

        #Update F_rhotau
        term_rho = 2 * F_sigmatau @ C_sigmarho + alpha * (Y_rho - U_rho).T
        F_rhotau= term_rho @ R_rho_inv
        
        #Update F_sigmatau
        term_sigma = 2 * F_rhotau @ C_rhosigma + alpha * (Y_sigma - U_sigma).T
        F_sigmatau = term_sigma @ R_sigma_inv
        
        #Update Y_rho
        Y_rho_tilde = F_rhotau.T + U_rho 
        Y_rho = proj_stiefel(Y_rho_tilde)
        
        #Update Y_sigma
        Y_sigma_tilde = F_sigmatau.T + U_sigma 
        Y_sigma = proj_stiefel(Y_sigma_tilde)
        
        #Update scaled dual variables
        U_rho +=  F_rhotau.T - Y_rho
        U_sigma += F_sigmatau.T - Y_sigma
        
        #Primal residuals
        r_p1 = norm(Y_rho - F_rhotau.T, 'fro')
        r_p2 = norm(Y_sigma - F_sigmatau.T, 'fro')
        
        #Dual residuals
        r_d1 = alpha * norm(Y_rho - Y_rho_prev, 'fro')
        r_d2 = alpha * norm(Y_sigma - Y_sigma_prev, 'fro')
        
        #Stopping criteria
        eps_p1 = tol_abs * anp.sqrt(n_rho * n_tau) + tol_rel * max(norm(Y_rho, 'fro'), norm(F_rhotau, 'fro'))
        eps_p2 = tol_abs * anp.sqrt(n_sigma * n_tau) + tol_rel * max(norm(Y_sigma, 'fro'), norm(F_sigmatau, 'fro'))
        eps_d1 = tol_abs * anp.sqrt(n_rho * n_tau) + tol_rel * alpha * norm(U_rho, 'fro')
        eps_d2 = tol_abs * anp.sqrt(n_sigma * n_tau) + tol_rel * alpha * norm(U_sigma, 'fro')
        
        #Check convergence
        if (r_p1 <= eps_p1 and r_p2 <= eps_p2 and r_d1 <= eps_d1 and r_d2 <= eps_d2):
            print(f"Converged at iteration {k+1}")
            return F_rhotau, F_sigmatau, Y_rho, Y_sigma, U_rho, U_sigma

    print("ADMM did not converge in {} iterations.".format(max_iter))
    
    return F_rhotau, F_sigmatau, Y_rho, Y_sigma, U_rho, U_sigma

def local_solver_ortoghonal(X_rho, X_sigma):
    """
    Solves the local (unweighted) problem by using the Procustes method, setting F_sigmatau=I_n, with n=n_tau=n_rho=n_sigma.
    
    INPUT
    =====
    X_rho : anp.ndarray, shape (n_rho, d). Data matrix associated with the node rho.
    X_sigma : anp.ndarray, shape (n_sigma, d). Data matrix associated with the node rho.
    seed: int, optional (default=42). Initialization seed.
        
    OUTPUT
    ======
    F_rhotau is an array of shape (n, n).
    """
    U, _, Vt = anp.linalg.svd(X_sigma@X_rho.T, full_matrices=False)
    
    return U@Vt, anp.eye(X_rho.shape[0])

def local_solver_unweighted(X_rho, X_sigma, n_tau, alpha=1., tol_abs=1.e-4, tol_rel=1.e-4, max_iter=1000, seed=42):
    """
    Solver for the local supbroblem:
    - In the orthogonal case, returns the solution in "Di Nino, L., Barbarossa, S., & Di Lorenzo, P. (2025). Learning Sheaf Laplacian Optimizing Restriction Maps."
    - In the general case, returns the solution via SOC: the transpose of the restriction maps belong to Stiefel manifold. 
    
    INPUT
    =====
    X_rho : anp.ndarray, shape (n_rho, d). Data matrix associated with the node rho.
    X_sigma : anp.ndarray, shape (n_sigma, d). Data matrix associated with the node sigma.
    n_tau : int. Dimensionality of the stalk on the edge.
    alpha : float, optional. ADMM stepsize (only used for the Stiefel solver).
    tol_abs : float, optional. Absolute tolerance for the stopping criterion (default=1.e-4).
    tol_rel : float, optional. Relative tolerance for the stopping criterion (default=1.e-4).
    max_iter : int, optional. Maximum number of ADMM iterations (default=1000).
    seed : int, optional. Initialization seed (default=42).
    
    OUTPUT
    ======
    Orthogonal case: F_rhotau, np.eye(n_tau)
    General case: F_rhotau, F_sigmatau, Y_rho, Y_sigma, U_rho, U_sigma : tuple of anp.ndarray
        Here:
          - F_rhotau is an array of shape (n_tau, n_rho)
          - F_sigmatau is an array of shape (n_tau, n_sigma)
          - Y_rho is an array of shape (n_rho, n_tau) 
          - Y_sigma is an array of shape (n_sigma, n_tau) 
          - U_rho and U_sigma are the scaled dual variables (same shape as Y_rho and Y_sigma, respectively).
    """
    n_rho, _ = X_rho.shape
    n_sigma, _ = X_sigma.shape
    
    if n_tau == n_rho == n_sigma:
        return local_solver_ortoghonal(X_rho, X_sigma)
    else:
        return local_solver_stiefel(X_rho, X_sigma, n_tau, alpha, tol_abs, tol_rel, max_iter, seed)