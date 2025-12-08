"""
solver.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np 
import cvxpy as cp
# import wandb 

from scipy.optimize import minimize
from tqdm import tqdm

# Riemannian optimization toolbox
import autograd.numpy as anp
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import SpecialOrthogonalGroup, Stiefel, Product
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions

#############################################################################################
#################### LINEAR OPERATORS FOR COMBINATORIAL LAPLACIANS ##########################
#############################################################################################

def L(
    w,
    V
    ):
    '''
    Linear operator mapping a vector of non-negative edge weights to a combinatorial graph Laplacian.

    Args:
        w (np.ndarray): Input edge weights vector of dimension V(V-1)/2.
        V (int): Number of nodes in the graph.

    Returns:
        L (np.ndarray): Laplacian-like structured V x V matrix.
    '''
    assert len(w) == (V * (V - 1)) // 2, "Invalid vector size for given dimension of the Laplacian"

    # Initialize a p x p matrix with zeros
    L = np.zeros((V, V))

    # Fill the upper triangular part 
    upper_indices = np.triu_indices(V, k=1)
    L[upper_indices] = - w

    # Make the matrix symmetric
    L += L.T  

    # Set the diagonal as the sum of off-diagonal elements in each row
    np.fill_diagonal(L, - L.sum(axis=1))

    return L

def L_adjoint(
    M: np.ndarray
    ) -> np.ndarray:
    '''
    Adjoint of the linear operator L(w).

    Args:
        M (np.ndarray): V x V matrix for which the adjoint must be computed.

    Returns:
        w (np.ndarray): Edges weights vector.
    '''

    N = M.shape[1]
    k = (N * (N - 1)) // 2
    j, l = 0, 1
    w = np.zeros(k)

    # The definition of the adjoint operator requires to define a moving index and compute the sum of 4 distinct entries of the matrix
    for i in range(k):
        w[i] = M[j, j] + M[l, l] - (M[l, j] + M[j, l])
        if l == (N - 1):
            j += 1
            l = j + 1
        else:
            l += 1

    return w

def L_inv(
    M
    ):
    '''
    Inverse of the linear operator L(w), mapping back a combinatorial graph Laplacian to the vector of edge weights.

    Args:
        M (np.ndarray): Input Laplacian-like matrix of dimension V x V.
    
    Returns:
        w (np.ndarray): Edges weights vector of dimension V(V-1)/2.
    '''
    N = M.shape[1]
    k = (N * (N - 1)) // 2
    w = np.empty(k)
    l = 0

    # Populate the vector from the upper triangular part of the input matrix
    for i in range(N - 1):
        for j in range(i + 1, N):
            w[l] = - M[i, j]
            l += 1
    
    # Check for non-negativity
    w[w < 0] = 0

    return w

#############################################################################################
###################### LINEAR OPERATORS FOR CONNECTION LAPLACIANS ###########################
#############################################################################################

def LKron(
    w,
    V,
    d
    ):
    '''
    Linear operator mapping a vector of non-negative edge weights to a d-Kronecker combinatorial Laplacian

    Args:
        w (np.ndarray): Edges weights vector of dimension V(V-1)/2.
        V (int): Number of nodes in the graph.
        d (int): Dimension of the stalks over the nodes.

    Returns:
        L_kron (np.ndarray): A d-Kronecker combinatorial Graph Laplacian of dimension dV x dV.
    '''

    L_kron = np.kron(L(w,V), np.eye(d))
    return L_kron

def LKron_adjoint(
        M: np.ndarray, 
        d: int) -> np.ndarray:
    '''
    Adjoint of the linear operator mapping edge weights to d-Kronecker combinatorial Laplacian

    Args:
        M (np.ndarray): dV x dV matrix for which the adjoint must be computed
        d (int): Dimension of the stalks over the nodes

    Returns:
        w (np.ndarray): Edges weights vectors of dimension V(V-1)/2
    '''
    N = M.shape[1] // d  
    k = (N * (N - 1)) // 2  
    j, l = 0, 1
    w = np.zeros(k)

    # Similarly to the combinatorial one, this operator requires the scanning of specific blocks of the connection Laplacians
    for i in range(k):
        block_jj = M[j*d:(j+1)*d, j*d:(j+1)*d]
        block_ll = M[l*d:(l+1)*d, l*d:(l+1)*d]
        block_lj = M[l*d:(l+1)*d, j*d:(j+1)*d]
        block_jl = M[j*d:(j+1)*d, l*d:(l+1)*d]

        w[i] = np.trace(block_jj + block_ll - block_lj - block_jl)

        if l == (N - 1):
            j += 1
            l = j + 1
        else:
            l += 1

    return w

def LKron_inv(  
    M,
    O,
    d
    ):
    '''
    Inverse of the linear operator L(w), mapping back any consistent connection graph Laplacian to the vector of edge weights or making least-square
    estimate for input matrices not possessing consistency requisite.

    Args:
        M (np.ndarray): Input Laplacian-like matrix of dimension V x V.
        O (np.ndarray): Block-diagonal matrix containing on each v-th block the orthonormal basis associated to the v-th node of the graph. 
        d (int): Dimensions of the stalks over the nodes.
    
    Returns:
        w (np.ndarray): Edges weights vector of dimension V(V-1)/2.
    '''

    N = int(M.shape[1] // d)
    k = (N * (N - 1)) // 2
    w = np.empty(k)
    l = 0

    # Recover the Kronecker graph laplacian
    LK = O @ M @ O.T

    # Populate the vector from the upper triangular part of the input matrix taking into account how the maps contribute to the Laplacian structure
    for i in range(N - 1):
        for j in range(i + 1, N):
            w[l] = - (1/d) * np.trace(LK[i * d : (i + 1) * d, j * d : (j + 1) * d])
            l += 1

    # Check for non-negativity
    w[w < 0] = 0
    return w

def L_spy(
    L,
    d
    ):
    '''
    Return the sparsity pattern of a laplacian-like matrix
    '''
    N = L.shape[1] // d
    k = (N * (N - 1)) // 2
    w = np.empty(k)
    l = 0

    # Populate the vector from the upper triangular part of the input matrix
    for i in range(N - 1):
        for j in range(i + 1, N):
            block = L[i * d : (i + 1) * d, j * d : (j + 1) * d]
            w[l] = int(not np.all(np.isclose(block, 0, atol=1e-8)))
            l += 1
    
    return w.astype(int)

#############################################################################################
#################### AKAIKE INFORMATION CRITERION KERNEL DETECTOR ###########################
#############################################################################################

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
    
    return Kmax - k_AIC


#############################################################################################
################### LEARNING ROUTINES - BLOCK-WISE UPDATE FUNCTIONS #########################
#############################################################################################

def Update_Z(
        w, 
        O,
        gamma,
        X,
        V,
        d
):
    
    LL_hat = O.T @ LKron(w = w, V = V, d = d) @ O
    # F = np.linalg.inv(gamma * np.eye(V * d) + LL_hat)
    F = np.linalg.inv(np.eye(V * d) + LL_hat / gamma)

    return F @ X

def Update_w(
    w, 
    U,
    S,
    O,
    lambda_,
    alpha,
    beta,
    gamma,
    V,
    d,
    its = 1,
    eps = 1e-8,
    proximal_mode = 'Proximal-L1',
    exact_linesearch = False
    ):  
    '''
    Optimization step in edge weights w in the learning problem in the form of a projected gradient descent coming from a MM approach. 

    Args:
        w (np.ndarray): Current iterate value for w.
        U (np.ndarray): Current iterate value for U (matrix of eigenvectors of the connection laplacian).
        S (np.ndarray): Emprical variance-covariance matrix of the given set of 0-cochains.
        O (np.ndarray): Current iterate value for O (block diagonal matrix of local node frames).
        lambda_ (np.ndarray): Current iterate value for lambda_ (variable controlling the spectrum of the inferred laplacians)-
        alpha (float): Hyperparameter controlling sparsity penalization
        beta (float): Hyperparameter controlling consistency and spectral regularization 
        V (int): Number of nodes in the graph.
        d (int): Dimensions of the bundle.
        its (int): Number of descent step to perform on w
        proximal_mode (bool): String for a proximal gradient descent given by the penalization term
        exact_linesearch (bool): Flag for the exact linesearch or the usage of the reciprocal of the operator norm as the stepsize

    Returns:
        w_hat (np.ndarray): Refined estimate of w
    '''

    assert proximal_mode in ['Proximal-L1','Proximal-LOG','ReweightedL1'], 'Please chose a valide proximal modality'

    S_hat = U @ (np.kron(np.diag(lambda_), np.eye(d))) @ U.T - (1 / (beta * gamma)) * O @ S @ O.T

    # Subroutine for positive l1 norm proximal projection
    def PositiveProxL1(x, th):
        return np.maximum(0, x - th)
    
    for _ in range(its):
        # The objective function in w is rewritten as a quadratic function in LKron(w) and linear in w
        # S_hat = O @ (U @ (np.kron(np.diag(lambda_), np.eye(d))) @ U.T - (1 / beta) * S ) @ O.T 

        # Gradients computation 
        grad = LKron_adjoint(LKron(w, V, d) - S_hat, d) 
        
        if exact_linesearch:
            # Armijo backtrack 
            c = LKron_adjoint(S_hat, d)
            L_kron_adg_grad = LKron_adjoint(LKron(grad, V, d), d)
            mu = np.dot(grad, L_kron_adg_grad) / (np.dot(w, L_kron_adg_grad) - np.dot(c, grad))
        else:
            # Operator norm is used as a proxy of the reciprocal of the stepsize
            mu = 2 * V * d

        if proximal_mode == 'Proximal-L1':

            w_hat = PositiveProxL1(w - (1 / mu) * grad, (alpha / beta) * 1 / (w + 1e-1))

        elif proximal_mode == 'Proximal-LOG':
            grad += (alpha / beta) * 1 / (w + eps)
            w_hat = np.maximum(0, w - (1 / mu) * grad)
        
        elif proximal_mode == 'ReweightedL1':
            w_hat = np.maximum(0, w - (1 / mu) * grad)

        w = w_hat
        
    return w

def Update_O_RG(
    O,
    S,
    U,
    w,
    V,
    d,
    eta,
    lambda_,
    beta,
    O_init = True,
    max_its=10,
    bases=None,
    solver = 'RCG',
    rho = 100,
    ):
    '''
    Optimization step in block diagonal matrix O collecting the nodes frames
    for the learning problem in the form of a Riemannian routine via pymanopt. 

    Args:
        O (np.ndarray): Current iterate value for O.
        S (np.ndarray): Emprical variance-covariance matrix of the given set of 0-cochains.
        U (np.ndarray): Current iterate value for U (matrix of eigenvectors of the connection laplacian).
        w (np.ndarray): Current iterate value for w (vectors of edge weights).
        V (int): Number of nodes in the graph.
        d (int): Dimension of the nodes stalks.
        eta (float): ? 
        lambda_ (np.ndarray): Current iterate value for lambda_ (variable controlling the spectrum of the inferred laplacians)-
        beta (float): Hyperparameter controlling consistency and spectral regularization 
        O_init (bool): Flag for whether the initialization of RCG is given by O or not.
        max_its (int): Maximum number of iterations of RCG subroutine
        bases (dict): Hashmap for an eventual prior knowledge on the nodes frames
        solver (str): Identifier for the Riemannian solver
    
    Returns:
        O_full (np.ndarray): Refined estimate of O
    '''

    assert solver in ['RCG', 'RSD', 'TR'], 'Invalid identifier for the Riemannian solver'

    # Matrix precomputation for computation streamline
    # A = S - beta * U @ np.kron(np.diag(lambda_), np.eye(d)) @ U.T 
    A = S 
    C = LKron(w, V, d)

    # Eventually inject prior knowledge in defining the manifold structure as a product one
    if bases is None:
        manifold = Product([Stiefel(d, d, retraction = 'polar') for _ in range(V)])
        # manifold = SpecialOrthogonalGroup(d, k = V, retraction = 'polar')
        bases = {}
        map_bases = {v: v for v in range(V)}
    else:
        if len(bases) == V:
            return O  
        
        free_vs = [v for v in range(V) if v not in bases]
        manifold = Product([Stiefel(d,d, retraction = 'polar') for _ in range(len(free_vs))])
        map_bases = {v: i for i, v in enumerate(free_vs)}

    # Subroutine for the cost function tr(OAO.TC) built by injecting the blkdiag structure 
    @pymanopt.function.autograd(manifold)
    def cost(*O_blocks):
        O = anp.zeros((d * V, d * V), dtype=anp.float64)
        for v in range(V):
            if v in bases.keys():
                O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(bases[v])
            else:
                if isinstance(O_blocks[map_bases[v]], anp.ndarray):
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]])
                else:
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]]._value)

        return anp.trace(O @ A @ O.T @ C) 
    
    # Subroutine for the euclidean gradient of tr(OAO.TC) built by injecting the blkdiag structure 
    @pymanopt.function.autograd(manifold)
    def euclidean_gradient(*O_blocks):
        O = anp.zeros((d * V, d * V), dtype=anp.float64)
        for v in range(V):
            if v in bases.keys():
                O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(bases[v])
            else:
                if isinstance(O_blocks[map_bases[v]], anp.ndarray):
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]])
                else:
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]]._value)

        gradients = []
        
        for v in range(V):
            if v not in bases.keys():
                grad_block = np.zeros((d, d), dtype=np.float64)
                
                grad_block += (
                    2 * 
                    C[v * d : (v + 1) * d, v * d : (v + 1) * d] @ 
                    O[v * d : (v + 1) * d, v * d : (v + 1) * d] @ 
                    A[v * d : (v + 1) * d, v * d : (v + 1) * d]
                )

                for m in range(V):
                    if m != v:
                        grad_block += ( 
                            C[v * d : (v + 1) * d, m * d : (m + 1) * d] @ 
                            O[m * d : (m + 1) * d, m * d : (m + 1) * d] @ 
                            A[m * d : (m + 1) * d, v * d : (v + 1) * d]
                            +
                            A[m * d : (m + 1) * d, v * d : (v + 1) * d] @ 
                            O[m * d : (m + 1) * d, m * d : (m + 1) * d].T @ 
                            C[v * d : (v + 1) * d, m * d : (m + 1) * d]
                            )
                
                gradients.append(grad_block)
        return gradients

    # Subroutine for the initialization (from O to list of variables)
    def O_spack(O, d, V):
        O_blocks = []
        for v in range(V):
            if v not in bases:
                O_blocks.append(O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d])
            else:
                O_blocks.append(bases[v])

        return O_blocks

    # Problem and solver istantation
    problem = Problem(manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    if solver == 'RCG':
        solver = ConjugateGradient(verbosity=0, max_iterations = max_its)
    elif solver == 'RSD':
        solver = SteepestDescent(verbosity=0, max_iterations = max_its)
    elif solver == 'TR':
        solver = TrustRegions(verbosity=0, max_iterations = max_its)

    if O_init:
        O_hat = solver.run(problem, initial_point = O_spack(O, d, V)).point
    else:
        O_hat = solver.run(problem, ).point

    # Reassemble full O
    O_full = anp.zeros((d * V, d * V))
    for v in range(V):
        if v in bases:
            O_full[v*d:(v+1)*d, v*d:(v+1)*d] = bases[v]
        else:
            O_full[v*d:(v+1)*d, v*d:(v+1)*d] = O_hat[map_bases[v]]

    return O_full

def Update_U(
    w,
    O,
    k,
    V,
    d,
    ):
    '''
    Optimization step in eigenvector matrix U in the form of a generalized Eigenvalue problem solution via
    Von Neuman trace inequality

    Args:
        w (np.ndarray): Current iterate value for w (vectors of edge weights).
        O (np.ndarray): Current iterate value for O (block diagonal matrix of local node frames).
        k (int): Number of connected components in the graph.
        V (int): Number of nodes in the graph.
        d (int): Dimension of the nodes stalks.
    
    Returns:
        U_hat (np.ndarray): Refined estimate of U
    '''

    # The solution is given directly by the eigenvectors of the estimated connection laplacian
    # LC_hat = O.T @ LKron(w, V, d) @ O
    LC_hat = LKron(w, V, d)
    _, Z = np.linalg.eigh(LC_hat)

    # Restriction to St(dV, d(V-k))
    U_hat = Z[:, k * d : ]
    return U_hat

def Update_O(
        O,
        Z,
        w,
        V, 
        d,
        rho,
        MAX_ITER = 1000,
        abs_tol = 1e-4,
        rel_tol = 1e-4
):
    
    # SOC approach to the block update in O over SO
    def BlockRetraction(M, V, d):
        P = np.zeros_like(M)
        for v in range(V):
            M_vv = M[v * d : (v + 1) * d , v * d : (v + 1) * d]
            U, _, Vt = np.linalg.svd(M_vv)
            P_vv_temp = U @ Vt
            if np.linalg.det(P_vv_temp) < 0:
                Sigma_tilde = np.ones(P_vv_temp.shape[0])
                Sigma_tilde[0] = np.linalg.det(P_vv_temp)
                P_vv = U @ Sigma_tilde @ Vt
                P_vv_temp = P_vv
            P[v * d : (v + 1) * d , v * d : (v + 1) * d] = P_vv_temp
        return P

    # Initialize support variables
    B = np.zeros_like(O)
    P = np.copy(O)
    n = V * d

    # Precomputation for fast inversion of Kronecker forms via spectral decomposition

    LambdaZ, UZ = np.linalg.eigh(Z @ Z.T / Z.shape[1])
    LambdaL, UL = np.linalg.eigh(LKron(w, V, d))

    LambdaZL = LambdaL[:, None] * LambdaZ[None, :] + rho

    for _ in range(MAX_ITER):

        # Store for loss control
        P_old = np.copy(P)

        # Primal updates
        O = UL @ ( (UL.T @ (P - B) @ UZ) * rho / LambdaZL ) @ UZ.T
        P = BlockRetraction(O + B, V, d)

        # Dual updates
        B += (O - P)
        
        # Compute residuals
        r_norm = np.linalg.norm(O - P, 'fro')
        s_norm = np.linalg.norm(rho * (P - P_old), 'fro')

        eps_pri = np.sqrt(n**2) * abs_tol + rel_tol * max(np.linalg.norm(O,'fro'), np.linalg.norm(P,'fro'))
        eps_dual = np.sqrt(n**2) * abs_tol + rel_tol * np.linalg.norm(rho * B, 'fro')

        # Check convergence
        if r_norm <= eps_pri and s_norm <= eps_dual:
            break
        
    return O


def Update_Lambda(
    U, 
    O,
    w, 
    beta, 
    gamma,
    c1, 
    c2,
    V,
    k,
    d
    ):

    '''
    Optimization step in the estimated eigenvalues in the form an isotonic regression over initial KKT guess.

    Args:
        U (np.ndarray): Current iterate value for U (matrix of eigenvectors of the connection laplacian).
        O (np.ndarray): Current iterate value for O (block diagonal matrix of local node frames).
        w (np.ndarray): Current iterate value for w (vectors of edge weights).
        beta (float): Hyperparameter controlling consistency and spectral regularization.
        c1 (float): Assumed lower bound on the eigenvalues.
        c2 (float): Assumed upper bound on the eigenvalues.
        V (int): Number of nodes in the graph.
        k (int): Number of connected components in the graph.
        d (int): Dimension of the nodes stalks.

    Returns:
        lambda_hat (np.ndarray): Refined estimate of lambda_
    '''

    # Compute initial estimate via KKT conditions
    # M = U.T @ O.T @ LKron(w, V, d) @ O @ U
    M = U.T @ LKron(w, V, d) @ U

    # Number of non zero eigenvalues
    q = V - k
    lambda_hat = np.zeros(q)

    D = np.zeros(q)
    for i in range(q):
        T = np.trace(M[i * d : (i + 1) * d, i * d : (i + 1) * d])
        D[i] = T
        lambda_hat[i] = 1 / (2 * d) * (T + np.sqrt(T ** 2 + (4 * d ** 2) / (gamma ** 2 * beta)))

    counter = 0

    # Isotonic heuristic (..., Palomar D., 2019)
    while not (
        np.all(lambda_hat >= c1)
        and 
        np.all(lambda_hat <= c2) 
        and
        np.all(lambda_hat[:-1] <= lambda_hat[1:])
    ):
        # Isotonic regression routine
        if counter < V - k + 1:
            # Enforce lower bound c1
            if np.any(lambda_hat < c1):
                r = np.max(np.where(lambda_hat < c1)[0])
                lambda_hat[:r + 1] = c1

            # Enforce upper bound c2
            if np.any(lambda_hat > c2):
                s = np.min(np.where(lambda_hat > c2)[0])
                lambda_hat[s:] = c2
            
            # Enforce non-decreasing order
            if np.any(lambda_hat[:-1] > lambda_hat[1:]):
                for i in range(q - 1):
                    if lambda_hat[i] > lambda_hat[i + 1]:
                        # Find m such that lambda[i:m] are decreasing
                        m = i
                        while m + 1 < q and lambda_hat[m] > lambda_hat[m + 1]:
                            m += 1

                        # Compute average d over i to m
                        d_avg = np.mean(D[i : m + 1])
                        
                        # Update lambda values
                        new_val = (1 / (2 * d)) * (d_avg + np.sqrt(d_avg ** 2 + (4 * d ** 2) / beta))
                        lambda_hat[i : m + 1] = new_val
                        break
            
            counter += 1

        else:
            return lambda_hat
        
    return lambda_hat

def Initialization(
    S, 
    d, 
    V,
    M,
    mode = 'ID',
    noisy = False,
    beta_0 = None,
    loss_tracker = False,
    bases = None,
    smooth=True,
    MAX_ITER = 1000,
    verbosity=0,
    reltol = 1e-4,
    abstol = 1e-6,
    seed = 42
    ):
    
    '''
    Initialization routine for w and O in the main loop.
    It can be performed according to different modalities:
        ID: w is initialized as a vector of all ones, O s.t. O_v = I_d for all v
        RANDOM: w is initialized as a vector of all ones, O s.t. O_v is in O(d) for all v
        NAIVE: Only possible if O is fully given, computes LKron_inv(OSO.T,d)
        QP: Jointly initalizes w and O solving ||pinv(S) - O.TLkron(w,d)O||^2 imposing the structure of both w and O
    
    Args:
        S (np.ndarray): Emprical variance-covariance matrix of the given set of 0-cochains.
        d (int): Dimension of the nodes stalks.
        V (int): Number of nodes in the graph.
        mode (str): Initialization mode.
        loss_tracker (bool): Flag for the log of initialization loss.
        bases (dict): Prior knowledge on the nodes bases.
        smooth (bool): Flag for the smoothness in the update in the QP subroutine
        MAX_ITER (int): Maximum number of iterations in the QP subroutine
        verbosity (str): ? 
        reltol (float): Relative tolerance on primal residuals to declare convergence in QP
        abstol (float): Absolute tolerance on primal residuals to declare convergence in QP
        seed (int): Random state
    '''
    
    # Seed for reproducibility
    np.random.seed(seed)

    # Mode assertion
    assert mode in ['ID','ID-QP','QP','NAIVE','RANDOM'], 'Invalid initialization modality'

    # Method to sample a matrix from SO(d)
    def RandomOn(
        ):
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            if np.linalg.det(Q) < 0:
                Q[0,:] *= - 1
            return Q
    
    # Constants for the diminishing stepsize
    l1 = 0.5
    l2 = 0.5
    gamma = 1

    # Pseudoinverse of the varcov matrix
    if noisy == False:
        S_pinv = np.linalg.pinv(S)
    else:
        if beta_0 is None:
            beta_0 = minAICdetector(S, V, d, )

        # Provide a noise free estimate of the pinv of the varcov matrix
        Lambda, U = np.linalg.eigh(S)

        # Consistent estimator of the noise variance
        sigma_2_hat = np.mean(Lambda[0 : d * beta_0])

        # Build a noise-free pseudoinverse of the varcov
        Lambda_hat = np.zeros_like(Lambda)
        Lambda_hat[d * beta_0 :] = 1 / (Lambda[d * beta_0 :] - sigma_2_hat)

        S_pinv = U @ np.diag(Lambda_hat) @ U.T

    # Subroutines for optimizing wrt w in QP initialization 
    def w_Objective(w, O):
        return 0.5 * np.linalg.norm(LKron(w, V, d) - O @ S_pinv @ O.T, 'fro') ** 2
    def w_grad(w, O):
        return LKron_adjoint(LKron(w, V, d) - O @ S_pinv @ O.T, d)
    
    # Subroutine for optimizing wrt O in QP initialization
    def O_Update(
        O_init,
        w,
        V,
        d,
        bases,
    ):
        C = LKron(w, V, d)

        if bases is None:
            manifold = Product([Stiefel(d, d, retraction = 'polar') for _ in range(V)])
            bases = {}
            map_bases = {v: v for v in range(V)}
        else:
            if len(bases) == V:
                return O  
            
            free_vs = [v for v in range(V) if v not in bases]
            manifold = Product([Stiefel(d, d, retraction = 'polar') for _ in range(len(free_vs))])
            map_bases = {v: i for i, v in enumerate(free_vs)}
        
        @pymanopt.function.autograd(manifold)
        def cost(*O_blocks):
            O = anp.zeros((d * V, d * V), dtype=anp.float64)
            for v in range(V):
                if v in bases.keys():
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(bases[v])
                else:
                    if isinstance(O_blocks[map_bases[v]], anp.ndarray):
                        O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]])
                    else:
                        O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]]._value)

            return - anp.trace(O @ S_pinv @ O.T @ C) 

        @pymanopt.function.autograd(manifold)
        def euclidean_gradient(*O_blocks):
            O = anp.zeros((d * V, d * V), dtype=anp.float64)
            for v in range(V):
                if v in bases.keys():
                    O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(bases[v])
                else:
                    if isinstance(O_blocks[map_bases[v]], anp.ndarray):
                        O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]])
                    else:
                        O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = anp.array(O_blocks[map_bases[v]]._value)

            gradients = []
            
            for v in range(V):
                if v not in bases.keys():
                    grad_block = np.zeros((d, d), dtype=np.float64)
                                        
                    grad_block -= (
                        2 * 
                        C[v * d : (v + 1) * d, v * d : (v + 1) * d] @ 
                        O[v * d : (v + 1) * d, v * d : (v + 1) * d] @ 
                        S_pinv[v * d : (v + 1) * d, v * d : (v + 1) * d]
                    )

                    for m in range(V):
                        if m != v:
                            grad_block -= (
                                C[v * d : (v + 1) * d, m * d : (m + 1) * d] @ 
                                O[m * d : (m + 1) * d, m * d : (m + 1) * d] @ 
                                S_pinv[m * d : (m + 1) * d, v * d : (v + 1) * d]
                                + 
                                S_pinv[m * d : (m + 1) * d, v * d : (v + 1) * d] @
                                O[m * d : (m + 1) * d, m * d : (m + 1) * d].T @ 
                                C[v * d : (v + 1) * d, m * d : (m + 1) * d]
                                )
                    
                    gradients.append( grad_block)
            return gradients

        # Problem and solver instantations
        problem = Problem(manifold, cost=cost, euclidean_gradient=euclidean_gradient)
        solver = ConjugateGradient(verbosity=0)
        O_hat = solver.run(problem, initial_point = O_init).point

        # Reassemble full O
        O_full = anp.zeros((d * V, d * V))
        for v in range(V):
            if v in bases:
                O_full[v*d:(v+1)*d, v*d:(v+1)*d] = bases[v]
            else:
                O_full[v*d:(v+1)*d, v*d:(v+1)*d] = O_hat[map_bases[v]]
        
        return O_full

    # Subroutine for initialization wrt O
    def O_spack(O, d, V, bases = bases):
        O_blocks = []
        if bases is None:
            bases = {}
        for v in range(V):
            if v not in bases:
                O_blocks.append(O[v*d:(v+1)*d, v*d:(v+1)*d])
        return O_blocks
    
    # Initialization of the initialization
    # Unitary weights, identity matrix

    w = np.ones(int(0.5 * (V - 1) * V))
    O = np.eye(d*V)

    if mode == 'ID' or mode == 'ID-QP':

        if bases is not None:
            for v in bases.keys():
                O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = bases[v]

        if mode == 'ID-QP':
            w = minimize(
                w_Objective, 
                w, 
                args=(O), 
                jac=w_grad, 
                method='L-BFGS-B', 
                bounds=[(0, None)] * len(w)).x
            if noisy:
                return w, O, sigma_2_hat
            else:
                return w, O
        else:    
            if noisy:
                return w, O, sigma_2_hat
            else:
                return w, O    
            
    if mode == 'RANDOM':
        for v in range(V):
            if bases is not None and v in bases.keys():
                O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = bases[v]
            else:
                O[v * d : ( v + 1 ) * d, v * d : ( v + 1 ) * d] = RandomOn()

        if noisy:
            return w, O, sigma_2_hat
        else:
            return w, O    
        
    if mode == 'QP':
        # Nonnegativity constraints on w
        bounds = [(0, None)] * len(w)
        loss = np.zeros(MAX_ITER)

        for iteration in tqdm(range(MAX_ITER)):
            # Optimization in w
            w_hat = minimize(
                w_Objective, 
                w, 
                args=(O), 
                jac=w_grad, 
                method='L-BFGS-B', 
                bounds=bounds).x

            # Optimization in O 
            O_hat = O_Update(
                O_spack(O, d, V, bases = bases),
                w, 
                V, 
                d,
                bases = bases,)

            # Convergence check
            loss[iteration] = np.linalg.norm(S_pinv - O.T @ LKron(w, V, d) @ O) ** 2
            
            w_err = np.abs(w - w_hat)
            O_err = np.abs(O - O_hat)

            converged_w = np.all(w_err <= 0.5 * reltol * (w + w_hat)) or np.all(w_err <= abstol)
            converged_O = np.all(O_err <= 0.5 * reltol * (O + O_hat)) or np.all(O_err <= abstol)

            if converged_w and converged_O:
                w = (1 - gamma) * w_hat + gamma * w
                O = (1 - gamma) * O_hat + gamma * O

                print(f'Initialization done in {iteration} iterations')
                break
            
            if smooth:
                w = w_hat
                O = O_hat

            else:
                w = (1 - gamma) * w + gamma * w_hat
                O = (1 - gamma) * O + gamma * O_hat

                # Stepsize update

                #gamma = (gamma + np.log(iteration + 1) ** l1) / (1 + l2 * np.sqrt(iteration + 1))
                gamma = (gamma + l1) / (1 + l2 * (iteration + 1))

                #print(gamma)

        if loss_tracker:
            return w, O, loss[:iteration]
        else:
            if noisy:
                return w, O, sigma_2_hat
            else:
                return w, O

#############################################################################################
################### LEARNING ROUTINES - MAIN SCGL OPTIMIZATION LOOP #########################
#############################################################################################

# Objective function 
def loss_(d, Z, X, V, gamma, lambda_, U, O, w, S, beta, alpha, noisy):
    if noisy:
        return (
            gamma * np.linalg.norm(X - Z) ** 2 / X.shape[1]
            - d * np.sum(np.log(lambda_))
            + np.trace(S @ O.T @ LKron(w, V, d) @ O) 
            + 0.5 * beta * np.linalg.norm(LKron(w, V, d) - U @ np.kron(np.diag(lambda_), np.eye(d)) @ U.T) ** 2
            + alpha * np.linalg.norm(w,ord=1)
        )
    else:
        return (
            - d * np.sum(np.log(lambda_))
            + np.trace(S @ O.T @ LKron(w, V, d) @ O) 
            + 0.5 * beta * np.linalg.norm(LKron(w, V, d) - U @ np.kron(np.diag(lambda_), np.eye(d)) @ U.T) ** 2
            + alpha * np.linalg.norm(w,ord=1)
        )

def SCGL_main_loop(
        w,
        U,
        S,
        X,
        Z,
        O,
        lambda_,
        MAX_ITER,
        alpha, 
        beta,
        gamma,
        noisy,
        V,
        d,
        max_w_its, 
        proximal_mode,
        exact_linesearch,
        update_frames,
        eta,
        beta_factor,
        max_O_its,
        SOC,
        bases,
        solver,
        c1,
        c2,
        k,
        eps,
        fix_beta,
        verbose,
        rho,
        beta_min,
        beta_max,
        reltol,
        abstol,
        loss_tol,
        patience,
        X_GT
        ):
    
    # Preallocating loss
    loss = np.zeros(MAX_ITER)
    S = Z @ Z.T / Z.shape[1]
    
    if proximal_mode != 'ReweightedL1':
        K = np.copy(S)
    else:
        H = alpha * (np.eye(d * V) - np.ones((d * V, d * V)))
        K = S + H

    # Initializing patience counter
    plateau_counter = 0

    # Handle initialization only baselines:
    if MAX_ITER == 0: 
        return O, w, None,
    else:
        for t in tqdm(range(MAX_ITER)):
            # Update Z
            if noisy:
                Z_hat = Update_Z(
                    w = w,
                    O = O,
                    gamma = gamma, 
                    X = X,
                    V = V,
                    d = d
                )

            else:
                Z_hat = X
     
            S = Z_hat @ Z_hat.T / Z_hat.shape[1]

            # Update w
            w_hat = Update_w(
                w = w, 
                U = U, 
                S = S, 
                O = O, 
                lambda_ = lambda_, 
                alpha = alpha, 
                beta = beta, 
                gamma=gamma,
                V = V, 
                d = d, 
                its = max_w_its, 
                proximal_mode = proximal_mode, 
                exact_linesearch = exact_linesearch
            )
            
            # Update O
            if update_frames == True:
                if not SOC:
                    O_hat = Update_O_RG(
                        O = O, 
                        S = S,
                        U = U, 
                        w = w_hat, 
                        V = V, 
                        d = d, 
                        eta = eta,
                        lambda_ = lambda_, 
                        beta = beta, 
                        O_init = True,
                        max_its = max_O_its,
                        bases = bases,
                        solver = solver
                    )
                else:
                    O_hat = Update_O(
                        O = O,
                        Z = Z_hat,
                        w = w_hat,
                        V = V,
                        d = d,
                        rho = rho,
                    )

            else:
                O_hat = O

            # Update U
            U_hat = Update_U(
                w = w_hat, 
                O = O_hat, 
                k = k, 
                V = V, 
                d = d,
            )

            # Update lambda
            lambda_hat = Update_Lambda(
                U = U_hat, 
                O = O_hat, 
                w = w_hat, 
                beta = beta, 
                gamma = gamma,
                c1 = c1, 
                c2 = c2, 
                V = V, 
                k = k, 
                d = d
            )

            if proximal_mode == 'ReweightedL1':
                K = S + H / (- O_hat.T @ LKron(w, V, d) @ O_hat + eps)
            else:
                K = S

            # Routine of scheduling for beta
            if not fix_beta:

                n_zero_eigenvalues = np.sum(np.isclose(np.abs(np.linalg.eigvals(LKron(w_hat, V, d))), 0, atol = 1e-9))
                if k * d < n_zero_eigenvalues:
                    if verbose:
                        print('Increasing beta...', beta)
                    beta *= (1 + beta_factor)

                elif k * d > n_zero_eigenvalues:
                    if verbose:
                        print('Decreasing beta...')
                    beta /= (1 + beta_factor)

                if beta < beta_min:
                    beta = beta_min

                if beta > beta_max:
                    beta = beta_max

            # Convergence check
            converged = True

            # Primal residuals on w
            w_err = np.abs(w - w_hat)
            converged_w = np.all(w_err <= 0.5 * reltol * (w + w_hat)) or np.all(w_err <= abstol)
            converged *= converged_w

            # Other residuals for check 
            # Z_err = np.abs(Z - Z_hat)
            # O_err = np.abs(O - O_hat)
            # U_err = np.abs(U - U_hat)
            # lambda_err = np.abs(lambda_ - lambda_hat)

            # print(np.linalg.norm(w_err), np.linalg.norm(Z_err), np.linalg.norm(O_err), )
            # print([np.linalg.norm(w_err), np.linalg.norm(Z_err),np.linalg.norm(O_err),np.linalg.norm(U_err),np.linalg.norm(lambda_err),])
            # # Primal residuals on O
            # if update_frames:
            #     O_err = np.abs(O - O_hat)
            #     converged_O = np.all(O_err <= 0.5 * reltol * (O + O_hat)) or np.all(O_err <= abstol)
            #     converged *= converged_O

            # # Primal residuals on U
            # U_err = np.abs(U - U_hat)
            # converged_U = np.all(U_err <= 0.5 * reltol * (U + U_hat)) or np.all(U_err <= abstol)
            # converged *= converged_U

            # # Primal residuals on lambda
            # lambda_err = np.abs(lambda_ - lambda_hat)
            # converged_lambd = np.all(lambda_err <= 0.5 * reltol * (lambda_ + lambda_hat)) or np.all(lambda_err <= abstol)
            # converged *= converged_lambd

            # Variables update
            w = w_hat
            Z = Z_hat

            if update_frames == True:
                O = O_hat

            U = U_hat
            lambda_ = lambda_hat

            if converged:

                print(f'Convergence reached in {t} iterations on the residuals')
                break

            # Loss storing 
            loss[t] = loss_(d, Z, X, V, gamma, lambda_, U, O, w, K, beta, alpha, noisy)
            if t > 0 :

                # Patience mechanism on loss plateaus
                relative_loss_change = np.abs(loss[t] - loss[t - 1]) / (np.abs(loss[t - 1]) + 1e-8)
                if relative_loss_change < loss_tol:
                    plateau_counter += 1
                    if plateau_counter >= patience:
    
                        print(f'Convergence assumed on the loss plateau at iteration {t}')
                        break
                else:
                    plateau_counter = 0

    return O, w, U, lambda_, loss[:t+1]

def SCGL(
    X,
    c1, 
    c2, 
    V, 
    k,
    d,
    eps,
    noisy = False,
    exact_linesearch = False,
    proximal_mode = 'Proximal-L1',
    verbose = False,
    initialization_mode = 'QP',
    initialization_seed = 42,
    w_inits = None,
    O_inits = None,
    bases = None,
    update_frames = True,
    alpha = None,
    beta = 20,
    eta = 1e-3,
    fix_beta = True,
    beta_min = 1,
    beta_max = 1e4,
    rho = 1e2,
    beta_factor = 5e-3,
    max_init_its = 1000,
    max_O_its = 10,
    SOC = False,
    solver = 'RCG',
    max_w_its = 1,
    MAX_ITER = 10000,
    patience = 100,
    reltol = 1e-4,
    abstol = 1e-5,
    loss_tol = 1e-4,
    full_report = False,
    X_GT = None
    ):
    '''
    Main optimization loop for the structured learning of consistent connection laplacians.

    Args:
        X (np.ndarray): Dataset of sampled 0-cochains
        S (np.ndarray): Sample covariance
        c1 (float): Assumed lower bound on the eigenvalues.
        c2 (float): Assumed upper bound on the eigenvalues.
        V (int): Number of nodes in the graph.
        k (int): Number of connected components in the graph.
        d (int): Dimension of the nodes stalks.
        exact_linesearch (bool): Flag for the exact linesearch or the usage of the reciprocal of the operator norm as the stepsize
        proximal_mode (str): Mode for a proximal gradient descent given by the penalization term or for a simpler gradient descent projected on the positive ortant.
        verbose (bool): Flag for verbosity in logging results.
        w_its (int): Number of descent step to perform on w.
        initialization_mode (str): Modality of initialization for w and O.
        initialization_seed (int): Random state for the initialization.
        w_inits (np.ndarray): Initialization for the weights
        O_inits (np.ndarray): Initialization for the nodes bases
        bases (dict): Prior knowledge on nodes frames.
        update_frames (bool): Flag to whether update or not the local basis on nodes.
        alpha (float): Hyperparameter controlling sparsity penalization.
        beta (float): Hyperparameter controlling consistency and spectral regularization.
        gamma (float): Hyperparameter for IIR filtering
        eta (float): ? 
        fix_beta (bool): Flag to whether modify or not beta during the optimization loop.
        beta_min (float): Minimum value for beta.
        beta_max (float): Maximum value for beta.
        rho (float): (1 + rho) is the multiplier for beta. 
        max_inits_its (int): Maximum number of iterations for the initialization.
        max_O_its (int): Maximum number of iterations on each sub-loop of RCG on O
        solver (str): Identifier for the riemannian solver in pymanopt
        max_w_its (int): Maximum number of iterations on each sub-loop of PGD on w
        MAX_ITER (int): Maximum number of iterations in the whole main loop
        patience (int): Parameter to early stop on loss plateaus
        reltol (float): Relative tolerance on primal residuals to declare convergence 
        abstol (float): Absolute tolerance on primal residuals to declare convergence 
        loss_tol (float): Tolerance on the loss improvement to trigger early stop after patience is saturated
        full_report (bool): Flag for the output
    Returns:
        O (np.ndarray): Block diagonal matrix of nodes frames
        w (np.ndarray): Vector of edge weights
        U (np.ndarray): Eigenvector basis of the connection laplacian
        lambda_ (np.ndarray): Eigenvalues of the connection laplacian
        loss (np.ndarray): Log of the loss curve
    '''
    
    # wandb.init(project="Structured Learning of Consistent Connection Graphs", config={
    #     "Scope": 'Topology and group action learning',
    #     "Network size": V,
    #     "Fiber size": d,
    #     "Prior assumption on number of connected components": k,
    #     "Are local frames learned?": update_frames,
    #     "Number of frames injected in the algorithm as prior knowledge": 0 if bases is None else len(bases.keys()),
    #     "Initial value for beta": beta,
    #     "Is beta fixed?": fix_beta,
    #     "Initial value for alpha": alpha,
    #     "Maximum number of iterations": MAX_ITER,
    #     "Initialization strategy": initialization_mode
    # })
    
    # # Finding best alpha if not given
    # if alpha is None:
    #     alpha = CValpha(X, c1, c2, beta, V, k, d, eps, folds, alphas_range_split, alphas_range)

    # # Embodying l1 penalization directly into the var-covar matrix
    # if not proximal:
    #     H = alpha * (np.eye(V*d) - np.ones((V*d,V*d)))
    #     K = S + H
    # else:
    #     K = S

    S = X @ X.T / X.shape[1]
    Z = np.copy(X)

    # Initialization of the block w
    if verbose:
        print('Initializing w and O...')

    if w_inits is None and O_inits is None:
        init_args = Initialization(
            M = X.shape[1],
            S = S, 
            d = d, 
            V = V,
            noisy = noisy,
            beta_0 = k, 
            mode = initialization_mode,
            MAX_ITER = max_init_its, 
            verbosity = verbose,
            bases = bases,
            seed = initialization_seed
            )
        
        w = init_args[0]
        O = init_args[1]

        if noisy:
            sigma_2_hat = init_args[2]
            gamma = 1 / (2 * sigma_2_hat)
        else:
            gamma = 1

    else:
        w = w_inits
        O = O_inits
        if noisy:
            sigma_2_hat = np.mean(np.linalg.eigvalsh(X @ X.T / X.shape[1])[0 : d * k])
            gamma = 1 / (2 * sigma_2_hat)
        else:
            gamma = 1

    if noisy:
        Z = Update_Z(w, O, gamma, X, V, d)

    # alpha = alpha / gamma
    # beta = beta / gamma
    
    # First call of each update method is to initialize the other blocks of variables
    if verbose:
        print('Initializing U...')

    U = Update_U(
        w = w,
        O = O,
        k = k,
        V = V,
        d = d        
    )

    if verbose:
        print('Initializing lambda...')

    lambda_ = Update_Lambda(
        U = U, 
        O = O, 
        w = w, 
        beta = beta, 
        gamma = gamma,
        c1 = c1, 
        c2 = c2, 
        V = V, 
        k = k, 
        d = d
        )
    
    # Main loop call
    O, w, U, lambda_, loss_log = SCGL_main_loop(        
        w = w,
        U = U,
        S = S,
        X = X,
        Z = Z,
        O = O,
        lambda_ = lambda_,
        MAX_ITER = MAX_ITER,
        alpha = alpha, 
        beta = beta,
        gamma = gamma,
        noisy = noisy,
        V = V,
        d = d,
        max_w_its = max_w_its, 
        proximal_mode = proximal_mode,
        exact_linesearch = exact_linesearch,
        update_frames = update_frames,
        eta = eta,
        max_O_its = max_O_its,
        bases = bases,
        solver = solver,
        c1 = c1,
        c2 = c2,
        k = k,
        eps = eps,
        fix_beta = fix_beta,
        verbose = verbose,
        beta_factor = beta_factor,
        rho = rho,
        SOC = SOC,
        beta_min = beta_min,
        beta_max = beta_max,
        reltol = reltol,
        abstol = abstol,
        loss_tol = loss_tol,
        patience = patience,
        X_GT = X_GT
        )
    
    #wandb.finish()
    if full_report:
        return O, w, Z, U, lambda_, loss_log
    else:
        return O, w, Z, loss_log

# # Hyperparameters tuning function
# def CValpha(X, c1, c2, beta, V, k, d, eps, folds = 5, alphas_range_split = 30, alphas_range = (0.01,0.5)):
#     def SampleCovariance(X):
#         X_mean = np.mean(X, axis=0)
#         X_centered = X - X_mean
#         S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
#         return S

#     print('Validating parameter alpha')

#     alphas = np.linspace(alphas_range[0], alphas_range[1], alphas_range_split)
#     m = X.shape[1]
#     KFold = np.array_split(np.arange(m), folds)
    
#     avg_errors = []

#     for alpha in alphas:
#         fold_errors = []
#         for idx in range(folds):
#             val_set = KFold[idx]
#             train_set = np.hstack([fold for i, fold in enumerate(KFold) if i != idx])

#             X_train = X[:, train_set]
#             X_val = X[:, val_set]

#             # Compute SCGL and validation error
#             O, w, U, lambda_, _ = SCGL(
#                 X_train, 
#                 SampleCovariance(X_train), 
#                 c1, 
#                 c2, 
#                 V, 
#                 k, 
#                 d, 
#                 eps, 
#                 alpha=alpha,
#                 beta=beta, 
#                 max_init_its=10, 
#                 MAX_ITER=100, 
#                 full_report=True
#                 )
            
#             val_error = loss_(d, V, lambda_, U, O, w, SampleCovariance(X_val), beta, alpha)
#             fold_errors.append(val_error)

#         # Save average error for current alpha
#         avg_errors.append(np.mean(fold_errors))
    
#     best_alpha = alphas[np.argmin(avg_errors)]
#     print(f"Best alpha via CV: {best_alpha}")
#     return best_alpha
