""" solver_utils.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""


import numpy as np 
import cvxpy as cp

from scipy.optimize import minimize
from tqdm import tqdm

import autograd.numpy as anp
import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel, Product
from pymanopt.optimizers import ConjugateGradient, SteepestDescent, TrustRegions

#############################################################################################
#################### LINEAR OPERATORS FOR COMBINATORIAL LAPLACIANS ##########################
#############################################################################################

def L(
    w : np.ndarray,
    V : int
) -> np.ndarray:
    """ Linear operator mapping a vector of non-negative edge weights to a combinatorial graph Laplacian.

    Parameters
    ----------
    w : np.ndarray)
        Input edge weights vector of dimension V(V-1)/2.
    V : int 
        Number of nodes in the graph.

    Returns
    -------
    np.ndarray
        Laplacian-like structured V x V matrix.
    """
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
    M : np.ndarray
) -> np.ndarray:
    """ Adjoint of the linear operator L(w).

    Parameters
    ----------
    M : np.ndarray
        V x V matrix for which the adjoint must be computed.

    Returns
    -------
    np.ndarray
        Edges weights vector.
    """

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
    M : np.ndarray
) -> np.ndarray:
    """ Inverse of the linear operator L(w), mapping back a combinatorial graph Laplacian to the vector of edge weights.

    Parameters
    ----------
    M : np.ndarray 
        Input Laplacian-like matrix of dimension V x V.
    
    Returns
    -------
    np.ndarray
        Edges weights vector of dimension V(V-1)/2.
    """
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
    w : np.ndarray,
    V : int,
    d : int
) -> np.ndarray:
    """ Linear operator mapping a vector of non-negative edge weights to a d-Kronecker combinatorial Laplacian

    Parameters
    ----------
    w : np.ndarray
        Edges weights vector of dimension V(V-1)/2.
    V : int
        Number of nodes in the graph.
    d : int
        Dimension of the stalks over the nodes.

    Returns
    -------
    np.ndarray 
        d-Kronecker combinatorial Graph Laplacian of dimension dV x dV.
    """

    L_kron = np.kron(L(w,V), np.eye(d))
    return L_kron


def LKron_adjoint(
    M: np.ndarray, 
    d: int
) -> np.ndarray:
    """ Adjoint of the linear operator mapping edge weights to d-Kronecker combinatorial Laplacian

    Parameters
    ----------
    M : np.ndarray
        dV x dV matrix for which the adjoint must be computed
    d : int
        Dimension of the stalks over the nodes

    Returns
    -------
    np.ndarray
        Edges weights vectors of dimension V(V-1)/2
    """
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


def L_spy(
    L : np.ndarray,
    d : int
) -> np.ndarray:
    """ Return the sparsity pattern of a laplacian-like matrix

    Parameters
    ----------
    L : np.ndarray
        Laplacian-like matrix
    d : int
        Stalk dimension

    Results
    -------
    np.ndarray
        Sparsity pattern in form of a binary vector for edges
    """
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


def Update_Z(
    w : np.ndarray, 
    O : np.ndarray,
    gamma : float,
    X : np.ndarray,
    V : int,
    d : int
) -> np.ndarray:
    """ Update step for Z in the learning problem in the form of a IIR filter

    Parameters
    ----------
    w : np.ndarray
        Edge weights of the graph
    O : np.ndarray
        Local node frames
    gamma : float
        Regularization parameter
    X : np.ndarray 
        Signals to be filtered (Vn, num_samples)
    V : int
        Number of nodes
    d : int
        Stalk dimension
    
    Returns
    -------
    np.ndarray  
        Filtered signals
    """
    
    LL_hat = O.T @ LKron(w = w, V = V, d = d) @ O
    F = np.linalg.inv(np.eye(V * d) + LL_hat / gamma)

    return F @ X


def Update_w(
    w : np.ndarray, 
    U : np.ndarray,
    S : np.ndarray,
    O : np.ndarray,
    lambda_ : np.ndarray,
    alpha : float,
    beta : float,
    gamma : float,
    V : int,
    d : int,
    its : int = 1,
    eps : float = 1e-8,
    proximal_mode : str = 'Proximal-L1',
    exact_linesearch : bool= False
    ) -> np.ndarray:  
    """ Optimization step in edge weights w in the learning problem in the form of a projected gradient descent coming from a MM approach. 

    Parameters
    ----------
        w : np.ndarray 
            Current iterate value for w.
        U : np.ndarray 
            Current iterate value for U (matrix of eigenvectors of the connection laplacian).
        S : np.ndarray 
            Emprical variance-covariance matrix of the given set of 0-cochains.
        O : np.ndarray 
            Current iterate value for O (block diagonal matrix of local node frames).
        lambda_ : np.ndarray 
            Current iterate value for lambda_ (variable controlling the spectrum of the inferred laplacians)-
        alpha : float 
            Hyperparameter controlling sparsity penalization
        beta : float 
            Hyperparameter controlling consistency and spectral regularization 
        V : int 
            Number of nodes in the graph.
        d : int 
            Dimensions of the bundle.
        its : int 
            Number of descent step to perform on w
        proximal_mode : bool 
            String for a proximal gradient descent given by the penalization term
        exact_linesearch : bool
            Flag for the exact linesearch or the usage of the reciprocal of the operator norm as the stepsize

    Returns
    -------
    np.ndarray
        Refined estimate of w
    """

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
    O : np.ndarray,
    S : np.ndarray,
    w : np.ndarray,
    V : int,
    d : int,
    O_init : bool = True,
    max_its : int = 10,
    bases : dict = None,
    solver : str = 'RCG',
    ) -> np.ndarray:
    """ Optimization step in block diagonal matrix O collecting the nodes frames for the learning problem in the form of a Riemannian routine via pymanopt. 

    Parameters
    ----------

        O : np.ndarray 
            Current iterate value for O.
        S : np.ndarray 
            Emprical variance-covariance matrix of the given set of 0-cochains.
        w : np.ndarray 
            Current iterate value for w (vectors of edge weights).
        V : int 
            Number of nodes in the graph.
        d : int 
            Dimension of the nodes stalks.
        O_init : bool 
            Flag for whether the initialization of RCG is given by O or not.
        max_its : int 
            Maximum number of iterations of RCG subroutine
        bases : dict 
            Hashmap for an eventual prior knowledge on the nodes frames
        solver : str 
            Identifier for the Riemannian solver
    
    Returns
    -------
    np.ndarray
        Refined estimate of O
    """

    assert solver in ['RCG', 'RSD', 'TR'], 'Invalid identifier for the Riemannian solver'

    # Matrix precomputation for computation streamline
    A = S 
    C = LKron(w, V, d)

    # Eventually inject prior knowledge in defining the manifold structure as a product one
    if bases is None:
        manifold = Product([Stiefel(d, d, retraction = 'polar') for _ in range(V)])
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


def Update_O_SOC(
    O,
    Z,
    w,
    V, 
    d,
    rho,
    MAX_ITER = 1000,
    abs_tol = 1e-4,
    rel_tol = 1e-4
) -> np.ndarray:
    """ Solves the O step with a Splitting Orthogonality Constraint approach 

    Parameters
    ----------
    O : np.ndarray
        Local node frames
    Z : np.ndarray
        Current signal estimate (Vn, n_samples)
    w : np.ndarray
        Current estimate of edge weights
    V : int
        Number of nodes
    d : int 
        Stalk dimension
    rho : float
        Parameter regulating convexity
    MAX_ITER : int
        Maximum number of iterations
    abs_tol : float
        Absolute error tolerance for convergence
    rel_tol : float
        Relative error tolerance for convergence
    
    Returns
    -------
    np.ndarray
        Refined estimate for O 
    """
    # SOC approach to the block update in O over SO
    def block_retraction(
        M : np.ndarray, 
        V : int, 
        d : int
    ) -> np.ndarray:
        """ Performs a block-wise polar factor retraction on SO

        Parameters
        ----------
        M : np.ndarray
            Block matrix to retracted
        V : int
            Number of nodes
        d : int
            Stalk dimension
        
        Returns
        -------
        np.ndarray
            Retracted block matrix
        """
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
        P = block_retraction(O + B, V, d)

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


def Update_U(
    w : np.ndarray,
    k : int,
    V : int,
    d : int,
) -> np.ndarray:
    """ Optimization step in eigenvector matrix U in the form of a generalized Eigenvalue problem solution via Von Neuman trace inequality

    Parameters
    ----------
    w : np.ndarray 
        Current iterate value for w (vectors of edge weights).
    k : int 
        Number of connected components in the graph.
    V : int 
        Number of nodes in the graph.
    d : int
        Dimension of the nodes stalks.
    
    Returns
    -------
    np.ndarray
        Refined estimate of U
    """

    # The solution is given directly by the eigenvectors of the estimated connection laplacian
    LC_hat = LKron(w, V, d)
    _, Z = np.linalg.eigh(LC_hat)

    # Restriction to St(dV, d(V-k))
    U_hat = Z[:, k * d : ]
    return U_hat


def Update_Lambda(
    U : np.ndarray, 
    w : np.ndarray, 
    beta : float, 
    gamma : float,
    c1 : float, 
    c2 : float,
    V : int,
    k : int,
    d : int
) -> np.ndarray:
    """ Optimization step in the estimated eigenvalues in the form an isotonic regression over initial KKT guess.

    Parameters
    ----------
        U : np.ndarray
            Current iterate value for U (matrix of eigenvectors of the connection laplacian).
        O : np.ndarray
            Current iterate value for O (block diagonal matrix of local node frames).
        w : np.ndarray
            Current iterate value for w (vectors of edge weights).
        beta : float
            Hyperparameter controlling consistency and spectral regularization.
        c1 : float
            Assumed lower bound on the eigenvalues.
        c2 : float
            Assumed upper bound on the eigenvalues.
        V : int
            Number of nodes in the graph.
        k : int
            Number of connected components in the graph.
        d : int
            Dimension of the nodes stalks.

    Returns
    -------
    np.ndarray 
        Refined estimate of lambda_
    """

    # Compute initial estimate via KKT conditions
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
    S : np.ndarray, 
    d : int, 
    V : int,
    mode : str = 'ID',
    noisy : bool = False,
    beta_0 : int = None,
    loss_tracker : bool = False,
    bases : dict= None,
    MAX_ITER : int = 1000,
    reltol : float = 1e-4,
    abstol : float = 1e-6,
    seed : int = 42
) -> None:
    """ Initialization routine for w and O in the main loop.
    It can be performed according to different modalities:
        ID: w is initialized as a vector of all ones, O s.t. O_v = I_d for all v
        RANDOM: w is initialized as a vector of all ones, O s.t. O_v is in O(d) for all v
        QP: Jointly initalizes w and O solving ||pinv(S) - O.TLkron(w,d)O||^2 imposing the structure of both w and O
        ID-QP: w is initialized solving ||pinv(S) - O.TLkron(w,d)O||^2 imposing the structure of w, while O_v = I_d for all v

    
    Parameters
    ----------
    S : np.ndarray 
        Emprical variance-covariance matrix of the given set of 0-cochains.
    d : int 
        Dimension of the nodes stalks.
    V : int 
        Number of nodes in the graph.
    mode : str 
        Initialization mode.
    loss_tracker : bool 
        Flag for the log of initialization loss.
    bases : dict 
        Prior knowledge on the nodes bases.
    smooth : bool 
        Flag for the smoothness in the update in the QP subroutine
    MAX_ITER : int 
        Maximum number of iterations in the QP subroutine
    reltol : float 
        Relative tolerance on primal residuals to declare convergence in QP
    abstol : float 
        Absolute tolerance on primal residuals to declare convergence in QP
    seed : int 
        Random state
    """
    
    # Seed for reproducibility
    np.random.seed(seed)

    # Mode assertion
    assert mode in ['ID','ID-QP','QP','RANDOM'], 'Invalid initialization modality'

    # Method to sample a matrix from SO(d)
    def RandomOn(
        ):
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            if np.linalg.det(Q) < 0:
                Q[0,:] *= - 1
            return Q

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
                w = w_hat
                O = O_hat

                print(f'Initialization done in {iteration} iterations')
                break
            
            w = w_hat
            O = O_hat

        if loss_tracker:
            return w, O, loss[:iteration]
        else:
            if noisy:
                return w, O, sigma_2_hat
            else:
                return w, O

def loss_(
    V : int,
    d : int, 
    X : np.ndarray, 
    Z : np.ndarray,
    U : np.ndarray, 
    O : np.ndarray,
    w : np.ndarray,
    S : np.ndarray, 
    lambda_ : np.ndarray,
    gamma : float,
    beta : float,
    alpha : float, 
    noisy : bool
) -> float:
    """ Computes the loss function value given the current state of the algorithm 

    Parameter
    ---------
    V : int
        Number of nodes
    d : int 
        Stalk dimension
    X : np.ndarray
        Observed signals
    Z : np.ndarray
        Currente estimate of denoised signals
    U : np.ndarray
        Current estimate of the eigenvectors of the laplacian
    O : np.ndarray
        Current estimate of the nodes basis
    w : np.ndarray
        Current estimate of the edge weights
    S : np.ndarray
        Current estimate of the covariance matrix
    lambda_ : np.ndarray
        Current estimate of the eigenvalues of the laplacian
    gamma : float
        Hyperparameter regulating the reconstruction error
    beta : float
        Hyperparameter regulating consistency
    alpha : float
        Hyperparameter regulating sparsity
    noisy : bool
        Flag for noise in learning
    
    Returns
    -------
    float
        Loss function value
    """
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
    
# def LKron_inv(  
#     M,
#     O,
#     d
#     ):
#     '''
#     Inverse of the linear operator L(w), mapping back any consistent connection graph Laplacian to the vector of edge weights or making least-square
#     estimate for input matrices not possessing consistency requisite.

#     Args:
#         M (np.ndarray): Input Laplacian-like matrix of dimension V x V.
#         O (np.ndarray): Block-diagonal matrix containing on each v-th block the orthonormal basis associated to the v-th node of the graph. 
#         d (int): Dimensions of the stalks over the nodes.
    
#     Returns:
#         w (np.ndarray): Edges weights vector of dimension V(V-1)/2.
#     '''

#     N = int(M.shape[1] // d)
#     k = (N * (N - 1)) // 2
#     w = np.empty(k)
#     l = 0

#     # Recover the Kronecker graph laplacian
#     LK = O @ M @ O.T

#     # Populate the vector from the upper triangular part of the input matrix taking into account how the maps contribute to the Laplacian structure
#     for i in range(N - 1):
#         for j in range(i + 1, N):
#             w[l] = - (1/d) * np.trace(LK[i * d : (i + 1) * d, j * d : (j + 1) * d])
#             l += 1

#     # Check for non-negativity
#     w[w < 0] = 0
#     return w