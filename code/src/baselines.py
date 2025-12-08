"""
baselines.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np 
import cvxpy as cp

from tqdm import tqdm

from itertools import product
from sklearn.model_selection import KFold
from scipy.fft import dct

###################################################################################################
# "Vector diffusion maps and the Connection Laplacian" (A.Singer,..., 2014)
# The following is an implementation of the two-steps connection laplacian inference framework 
# based on the approximation of the related continuous operator on Riemannian Manifolds
###################################################################################################

class VectorDiffusionMaps:
    '''
    This class implements the estimation of the connection laplacian via local PCA and vector diffusion maps allignment over a point cloud.
    Args:
        X (np.ndarray): The point cloud dataset with dimension (manifold_dim, number_of_points)
        epsilon_geometric (float): Hyperparameter controlling the underpinning geometric graph sparsity
        epsilon_PCA (float): Hyperparameter controlloing the PCA approximation
        gamma (float): Hyperparameter controlling the embedded dimension estimation
        mode (str): Kernel function mode
    Methods:
        Kernel -> Kernel function applied to euclidean distances
        PairwiseDistances -> Preocompute the matrix of distances with the Gram trick 
        LocalPCA -> Compute the local frame for a node v as polar factors of the covariance matrix
        LocalPCA_full -> Compute the local frame for all nodes
        Alignment ->  Compute a geometric graph and a connection laplacian over it via Procrustes alignment
    '''
    def __init__ (
            self,
            X,
            epsilon_geometric,
            epsilon_PCA,
            gamma = 0.9,
            mode = 'Gaussian'
    ):
        # The dataset of points from a Riemannian manifold and its dimensions
        self.X = X
        self.V = X.shape[1]
        self.d = X.shape[0]

        # Thresholds for the geometric construction, the local basis approximation and the dimension estimation
        self.epsilon_geometric = epsilon_geometric
        self.epsilon_PCA = epsilon_PCA
        self.gamma = gamma

        # Kernel mode
        self.mode = mode

        # Pairwise distances
        self.dist_matrix = self.PairWiseDistances()

        # Connection Laplacian construction
        self.LocalPCA_Full()
        self.Alignment()

    def Kernel(
            self,
            Z,
    ):
        '''
        Kernel function applied to the euclidean distances
        '''
        supp = np.array((Z <= 1) * (0 <= Z), dtype =float)
        if self.mode == 'Gaussian':
            return np.exp(- Z ** 2) * supp
        elif self.mode == 'Epanechnikov':
            return (1 - Z ** 2) * supp
    
    def PairWiseDistances(
            self
    ):
        '''
        Compute the matrix of pairwise distances between all points using the Gram trick
        '''
        G = self.X.T @ self.X
        S = np.diag(G)
        return np.sqrt(np.maximum(S[:,None] + S[None,:] - 2 * G, 0))
    
    def LocalPCA(
            self,
            v
    ):
        '''
        Compute the local representation frame of each node as the polar factor of the covariance matrix
        '''

        # Point coordinates at node v
        x_v = self.X[:,v]

        # Neighborhood of x_v
        N_v = (0 < self.dist_matrix[v, :]) * (self.dist_matrix[v, :] <= np.sqrt(self.epsilon_PCA))

        # Restricting X to x_v and centering
        X_ = self.X[:,N_v] - x_v[:, np.newaxis]

        # Computing the diagonal scaling matrix C
        scaled_distances = self.dist_matrix[v, N_v] / np.sqrt(self.epsilon_PCA)
        kernelized = np.sqrt(self.Kernel(scaled_distances))
        C = np.diag(kernelized)

        # Computing the local basis
        B = X_ @ C
        M, Sigma, _ = np.linalg.svd(B)

        # Local dimension estimate
        Sigma2 = Sigma ** 2
        cumsum_normalized = np.cumsum(Sigma2) / np.sum(Sigma2)
        d = np.searchsorted(cumsum_normalized, self.gamma, side='right')

        return M, d
    
    def LocalPCA_Full(
            self
    ):
        '''
        Applying the LocalPCA routine to all the points in the dataset
        '''
        
        self.d_hats = np.zeros(self.V)      # Preallocating for estimation of the embedded dimension 
        self.local_basis = {}               # Preallocating to store local frames

        for v in range(self.V):
            M, d = self.LocalPCA(v)
            self.local_basis[v] = M
            self.d_hats[v] = d
        
        # Estimating manifold embedded dimension and using it to restrict the local basis to d_hat-dimensional span
        self.d_hat = int(np.ceil(np.median(self.d_hats)))
        self.local_basis = {v:self.local_basis[v][:,0:self.d_hat] for v in range(self.V)}

    def Alignment(
            self
    ):
        '''
        Compute a geometric graph and a connection laplacian over it via Procrustes alignment
        '''
        # Compute a geometric graph in terms of weights matrix with a gaussian Kernel
        self.W = self.Kernel(self.dist_matrix / np.sqrt(self.epsilon_geometric))

        # Compute the Procrustes transformation and some accessory stuff
        self.maps = {}
        self.degree_matrices = {}
        self.ndegree_matrices = {}

        self.degrees = np.sum(self.W, axis = 1)
        self.ndegrees = np.zeros(self.V)
        self.D = np.zeros((self.V * self.d_hat, self.V * self.d_hat))
        self.D_tilde = {}

        for v in range(self.V):
            self.ndegrees[v] = (
                1 / (self.degrees[v] + 1e-8) * 
                np.sum(self.W[:,v] / (self.degrees + 1e-8) )
                )
            
            self.degree_matrices[v] = self.degrees[v] * np.eye(self.d_hat)
            self.ndegree_matrices[v] = self.ndegrees[v] * np.eye(self.d_hat)

            self.D[ v * self.d_hat : (v + 1) * self.d_hat, v * self.d_hat : (v + 1) * self.d_hat] = self.ndegree_matrices[v]
            self.D_tilde[v] = np.linalg.inv(self.degree_matrices[v])

            for w in range(v + 1, self.V):
                if self.W[v, w] > 0:

                    O_vw_tilde = self.local_basis[v].T @ self.local_basis[w]
                    U, _, Vt = np.linalg.svd(O_vw_tilde)
                    O_vw = U @ Vt

                    self.maps[(v,w)] = self.W[v,w] * self.D_tilde[v] @ O_vw @ self.D_tilde[v]

        # Computing the normalized connection Laplacian
        self.S = np.zeros((self.V * self.d_hat, self.V * self.d_hat))
    
        for edge in self.maps.keys():
            i = edge[0]
            j = edge[1]

            self.S[i * self.d_hat : ( i + 1 ) * self.d_hat, j * self.d_hat : ( j + 1 ) * self.d_hat] = self.maps[edge]
            self.S[j * self.d_hat : ( j + 1 ) * self.d_hat, i * self.d_hat : ( i + 1 ) * self.d_hat] = self.maps[edge].T

        self.CL = (1 / self.epsilon_geometric) * (np.linalg.inv(self.D) @ self.S - np.eye(self.V * self.d_hat))

        return self.CL
    
###################################################################################################
# "Learning Sheaf Laplacian from Smooth Signals" (J.Hansen, R.Ghrist, 2019)
# The following is an implementation of Jacob Hansen framework for the learning of sheaf laplacians
# as minimum total variations problems over proper convex cones for specific sheaf-like structure
###################################################################################################

class SheafConnectionLaplacian:

    def __init__(
            self, 
            V, 
            d, 
            alpha=1.0, 
            beta=1.0, 
            gamma = 1.0,
            epsilon=1e-8, 
            L0=None
            ):
        
        self.V = V
        self.d = d
        self.L0 = L0
        self.epsilon = epsilon

        # Parameter for covariance matrix
        self.C = cp.Parameter((self.d * self.V, self.d * self.V), PSD=True)

        # Hyperparameters
        self.alpha = cp.Parameter(nonneg=True, value=alpha)
        self.beta  = cp.Parameter(nonneg=True, value=beta)
        self.gamma = cp.Parameter(nonneg=True, value=gamma)

        # Full edge list (upper triangular)
        self.edges = [(i,j) for i in range(V) for j in range(i+1,V)]
        self.E = len(self.edges)

        # Incidence matrix
        self.incidence = np.zeros((V, self.E))
        for e,(i,j) in enumerate(self.edges):
            self.incidence[i,e] = 1
            self.incidence[j,e] = 1

        # Build the problem
        self._precompute_index_maps()
        self._build_problem()

    def _precompute_index_maps(self):

        self.C_diag_param = cp.Parameter(self.E)              
        self.C_off_param  = cp.Parameter((self.E, self.d * self.d))  

    def _build_problem(self):
        # Variables initalization
        self.rho = cp.Variable(self.E, nonneg=True, name="rho")
        self.F   = cp.Variable((self.E, self.d*self.d), name="F")  

        # Conic constraints
        constraints = [self.rho >= cp.sqrt(self.d) * cp.norm(self.F, axis =1)]  

        # Volume-preserving constraint
        constraints += [cp.sum(self.rho) == self.gamma]

        # Likelihood term
        likelihood = self.rho @ self.C_diag_param + 2 * cp.sum(cp.multiply(self.F, self.C_off_param))

        # R1: log barrier on node degrees
        degrees = (self.incidence @ self.rho)
        R1 = - self.alpha * cp.sum(cp.log(self.d * degrees + self.epsilon))

        # R2: Frobenius norm on F
        R2 = self.beta * cp.sum_squares(self.F)

        # Objective function and problem istantiation
        objective = cp.Minimize(likelihood + R1 + R2)
        self.problem = cp.Problem(objective, constraints)

    def Solve(
            self, 
            C, 
            verbose=0, 
            solver=cp.MOSEK, 
            warm_start=False):
        
        # Fill parameters (outside CVXPY graph)
        C_diag_vals = []
        C_off_vals  = []
        for (i,j) in self.edges:
            Ci = C[ i * self.d : ( i + 1 ) * self.d, i * self.d : ( i + 1 ) * self.d]
            Cj = C[ j * self.d : ( j + 1 ) * self.d, j * self.d : ( j + 1 ) * self.d]
            Cij = C[ i * self.d : ( i + 1 ) * self.d, j * self.d : ( j + 1 ) * self.d]
            C_diag_vals.append(np.trace(Ci) + np.trace(Cj))
            C_off_vals.append(Cij.reshape(-1))

        self.C_diag_param.value = np.array(C_diag_vals)
        self.C_off_param.value  = np.vstack(C_off_vals)

        # Solve the problem
        self.problem.solve(solver=solver, verbose=verbose, warm_start=warm_start)
        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Solver failed with status: {self.problem.status}")

        # Explicit reconstruction of the Sheaf Laplacian
        blocks = [[np.zeros((self.d,self.d)) for _ in range(self.V)] for _ in range(self.V)]
        rho_val = self.rho.value
        F_val   = self.F.value.reshape((self.E, self.d, self.d))

        for e,(i,j) in enumerate(self.edges):
            blocks[i][j] = F_val[e]
            blocks[j][i] = F_val[e].T
            blocks[i][i] += rho_val[e] * np.eye(self.d)
            blocks[j][j] += rho_val[e] * np.eye(self.d)

        L_hat = np.block(blocks)
        L_hat[np.isclose(L_hat, 0, atol=1e-7)] = 0.0

        if self.L0 is None:
            return L_hat
        else:
            scale = (np.trace(L_hat.T @ self.L0) /
                     (np.linalg.norm(L_hat,'fro')**2 + 1e-12))
            return scale * L_hat
    
    def CrossValidation(
        self,
        X,
        alpha_beta_ratio=1.0,
        grid=None,
        k_folds=5,
        verbose=0,
        solver=cp.MOSEK
    ):
        """
        Cross-validation to tune beta, keeping alpha/beta constant.
        """
        # Scaling heuristic
        base_scale = (1 / X.shape[1]) * np.trace(X @ X.T)
        scale = base_scale / (self.V * self.d)

        if grid is None:
            beta_grid = [x * scale for x in [1e-3, 1e-2, 1e-1, 1.0, 1e1]]
            gamma_grid = [x * scale for x in [1e-2, 1e-1, 1.0, 1e1, 1e2]]

        best_score = float("inf")
        best_params = {"alpha": None, "beta": None}

        print("Validating best beta with fixed alpha/beta ratio...")

        for beta_val, gamma_val in product(beta_grid, gamma_grid):
            # enforce ratio alpha / beta = alpha_beta_ratio
            self.beta.value = beta_val
            self.alpha.value = alpha_beta_ratio * beta_val
            self.gamma.value = gamma_val

            fold_scores = []
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            for train_idx, val_idx in kf.split(X.T):
                X_train = X[:, train_idx]
                X_val   = X[:, val_idx]

                # covariance matrices
                C_train = (1 / X_train.shape[1]) * (X_train @ X_train.T)
                C_val   = (1 / X_val.shape[1]) * (X_val @ X_val.T)

                try:
                    # fit on training covariance
                    L_hat = self.Solve(C_train, verbose=verbose, solver=solver, warm_start=True)
                    #L_hat = L_hat / np.linalg.norm(L_hat,)

                except Exception as e:
                    if verbose:
                        print(f"Solver failed on fold: {e}")
                    fold_scores.append(np.inf)
                    continue

                # Validation score
                if self.L0 is None:
                    likelihood = np.trace(L_hat @ C_val)
                    diag_blocks_sum = np.array([
                        np.linalg.det(L_hat[i*self.d:(i+1)*self.d, i*self.d:(i+1)*self.d])
                        for i in range(self.V)
                    ])
                    R1 = - self.alpha.value * np.sum(np.log(diag_blocks_sum + self.epsilon))
                    R2 = self.beta.value * np.sum(self.F.value ** 2)
                    score = likelihood + R1 + R2
                else:
                    # ground truth Laplacian available
                    score = np.linalg.norm(L_hat - self.L0, ord=1)

                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            print(f"[CV] beta = {beta_val}, alpha = {self.alpha.value}, avg_score = {avg_score:.4f}")

            if avg_score < best_score:
                best_score = avg_score
                best_params["beta"] = beta_val
                best_params["alpha"] = self.alpha.value

        print("Best beta/alpha validated")
        self.beta.value = best_params["beta"]
        self.alpha.value = best_params["alpha"]
        return best_params


###################################################################################################
# "Learning Sheaf Laplacian optimizing restriction maps" (L.Di Nino, et al.,  2024)
# The following is an implementation of our previous work on smooth learning with a prior 
# on the geometry of the restriction maps
###################################################################################################

class SmoothSheafDiffusion:
    def __init__(
        self,
        X,
        V,
        d,
        n_edges
    ):
        # Dataset of 0-cochains (dV, n_samples)
        self.X = X

        # Problem dimensions
        self.V = V
        self.d = d

        self.n_samples = self.X.shape[1]

        # Number of edges is required as prior knowledge
        self.n_edges = n_edges
    
        # Global shared dictionary is a Discrete Cosine Transform orthonormal basis
        self.basis = dct(np.eye(self.d), axis=0, norm='ortho').T

        # Two-steps inference
        self.best_alpha = self.CrossValidation()
        self.LocalSparsifying(self.best_alpha)
        self.Alignment()
    
    def CrossValidation(self, k_folds = 5):

        # Heuristics for the hyperparameters grid
        base_scale = (1 / self.X.shape[1]) * np.trace(self.X @ self.X.T)

        scale = base_scale / (self.V * self.d)
        alpha_grid = [x * scale for x in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]]

        best_score = float('inf')
        best_params = {'alpha': None}

        print('Validating best hyperparamters...')
        for alpha in alpha_grid:

            fold_scores = []
            fold_scores = []
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

            # Train model on training set
            for train_idx, val_idx in kf.split(self.X.T):
                
                X_train = self.X[:, train_idx]
                X_val = self.X[:, val_idx]

                C_val = (1 / X_val.shape[1]) * X_val @ X_val.T

                try:
                    self.LocalSparsifying(alpha, X_train)
                    self.Alignment()
                    L_hat = self.LaplacianBuilder()

                except Exception as e:
                    print(f"Solver failed for alpha={alpha}: {e}")
                    fold_scores.append(np.inf)
                    continue

                # Evaluate on validation set
                score = np.trace(L_hat @ C_val)
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            print(f"[CV] alpha={alpha},  avg_score={avg_score}")

            if avg_score < best_score:
                best_score = avg_score
                best_params = {'alpha': alpha}

        print('Best hyperparameters validated')

        return best_params['alpha']

    def LocalSparsifying(self, alpha, X = None):
        # Defining sub-routines for local sparsification
        def prox21_col(x, alpha):
            return ( 1 - alpha / (np.max([np.linalg.norm(x), alpha])) ) * x
        
        def prox21(X, alpha,):
            X = np.apply_along_axis(prox21_col, axis = 1, arr = X, alpha = alpha)
            return X

        def ProxGradDescent(X, alpha, LR = 3e-3, MAXITER = 1000, eps = 1e-3):
            S = np.zeros((self.d, X.shape[1]))
            loss = np.inf

            for _ in range(MAXITER):
                # Gradient descent
                grad = self.basis.T @ self.basis @ S - self.basis.T @ X
                S = S - LR * grad

                # Soft thresholding
                S = prox21(S, alpha)

                temp = np.linalg.norm(X - self.basis @ S)
                if loss - temp < eps:
                    break
                else:
                    loss = temp
            return S
        
        if X is None:
            X = self.X

        self.local_codes = {v: None for v in range(self.V)}
        self.local_bases = {v: None for v in range(self.V)}

        for v in range(self.V):
            # Perform local sparsification on the given shared basis
            X_v = X[v * self.d : ( v + 1 ) * self.d, :]
            self.local_codes[v] = ProxGradDescent(X_v, alpha)
            self.local_bases[v] = self.basis[:, np.abs(self.local_codes[v][:,0]) > 1e-8]

    def Alignment(self):
        self.maps = {
            (i,j): {
                i: None,
                j: None,
                }
                for i in range(self.V) for j in range(i + 1, self.V)
            }
        
        self.dists = {
            (i,j): 0
                for i in range(self.V) for j in range(i + 1, self.V)
            }
        
        for i in range(self.V):
            for j in range( i + 1, self.V):
                S_i = self.local_codes[i]
                S_j = self.local_codes[j]
                C_ij = S_i @ S_j.T / self.n_samples

                X, _, Y = np.linalg.svd(self.basis @ C_ij @ self.basis.T, full_matrices=False)
                F_i = Y.T @ X.T
                F_j = np.eye(self.d)
                self.maps[(i,j)][i] = F_i
                self.maps[(i,j)][j] = F_j

                self.dists[(i,j)] = np.linalg.norm(F_i @ self.basis @ S_i - self.basis @ S_j)

    def LaplacianBuilder(self):

        edges = list(sorted(self.dists.items(), key=lambda x:x[1]))[ : self.n_edges]
        L = np.zeros((self.d * self.V, self.d * self.V))

        for edge in edges: 
            u = edge[0][0]
            v = edge[0][1] 

            L[u * self.d : ( u + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] += np.eye(self.d)
            L[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] = - self.maps[(u,v)][u]

        return L 
    