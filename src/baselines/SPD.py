"""SDP.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

from itertools import product

import numpy as np 
import cvxpy as cp
from sklearn.model_selection import KFold

class SheafConnectionLaplacian:
    """Implementation of a conic programming learning algorithm from
    'Learning Sheaf Laplacian from Smooth Signals' (J.Hansen, R.Ghrist, 2019)
    restricted to a cone containing as a submanifold the O(n) group

    Parameters
    ----------
    V : int
        Number of nodes in the network
    d : int
        Stalk dimension
    alpha : float
        Hyperparameter governing the log-barreer
    beta : float 
        Hyperparameter governing the Ridge term
    gamma : float
        Hyperparameter governing the volume preserving constraint
    epsilon : float
        Control term to avoid by-zero divisions
    L0 : np.ndarray
        Ground truth Sheaf Laplacian
    """

    def __init__(
        self, 
        V : int, 
        d : int, 
        alpha : float = 1.0, 
        beta : float = 1.0, 
        gamma : float = 1.0,
        epsilon : float = 1e-8, 
        L0 : np.ndarray = None
    ) -> None:
        
        # Constructive parameter
        self.V = V
        self.d = d
        self.L0 = L0
        self.epsilon = epsilon

        # Parameter for covariance matrix
        self.C = cp.Parameter((self.d * self.V, self.d * self.V), PSD=True)

        # Hyperparameters of the problem
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
        self.precompute_index_maps()
        self.build_problem()

    def precompute_index_maps(self):
        """Precomputes index of entries of the sheaf laplacian to streamline computations
        """
        self.C_diag_param = cp.Parameter(self.E)              
        self.C_off_param  = cp.Parameter((self.E, self.d * self.d))  

    def build_problem(self):
        """Istantiates the learning problem
        """
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

    def solve(
        self, 
        C : np.ndarray, 
        verbose : int = 0, 
        solver : str = "MOSEK", 
        warm_start : bool = False
    ) -> np.ndarray:
        """Method to solve the istantiated problem

        Parameters
        ----------
        C : np.ndarray
            Empirical covariance
        verbose : int
            Flag for the level of verbosity of the procedure
        solver : str
            Label for the solver in cvxpy
        warm_start : bool
            Flag for the warm start in solving the problem
        
        Returns
        -------
        np.ndarray
            The estimated sheaf Laplacian
        """
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
            # L0 is given to minimize the distance from it via a proper scaling
            scale = (np.trace(L_hat.T @ self.L0) / (np.linalg.norm(L_hat,'fro')**2 + 1e-12))
            return scale * L_hat
    
    def cross_validation(
        self,
        X,
        alpha_beta_ratio : float = 1.0,
        k_folds : int = 5,
        verbose : int = 0,
        solver : str = "MOSEK"
    ):
        """Cross-validation to tune beta and gamma, keeping alpha/beta constant.

        Parameters
        ----------
        X : np.ndarray
            Input data as (V * d, n_samples) to compute foldable covariances
        alpha_beta_ratio : float
            Ratio between alpha and beta (default is 1)
        k_folds : int
            Number of folds in the validation procedure
        verbose : int
            Flag for the level of verbosity of the procedure
        solver : str
            Label for the solver in cvxpy
        
        Returns
        -------
        dict
            Table of the best alpha, beta, gamma parameters
        """
        # Scaling heuristic
        base_scale = (1 / X.shape[1]) * np.trace(X @ X.T)
        scale = base_scale / (self.V * self.d)

        beta_grid = [x * scale for x in [1e-3, 1e-2, 1e-1, 1.0, 1e1]]
        gamma_grid = [x * scale for x in [1e-2, 1e-1, 1.0, 1e1, 1e2]]

        best_score = float("inf")
        best_params = {
            "alpha": None, 
            "beta": None, 
            "gamma" : None}

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
                    # Fit on training covariance
                    L_hat = self.solve(C_train, verbose=verbose, solver=solver, warm_start=True)

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
                    # Ground truth Laplacian available
                    score = np.linalg.norm(L_hat - self.L0, ord=1)

                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            if verbose > 0:
                print(f"[CV] beta = {beta_val}, alpha = {self.alpha.value}, gamma = {self.gamma.value}, avg_score = {avg_score:.4f}")

            if avg_score < best_score:
                best_score = avg_score
                best_params["beta"] = self.beta.value
                best_params["alpha"] = self.alpha.value
                best_params["gamma"] = self.gamma.value

        print("Best beta/alpha validated")
        self.beta.value = best_params["beta"]
        self.alpha.value = best_params["alpha"]
        self.gamma.value = best_params["gamma"]

        return best_params
    