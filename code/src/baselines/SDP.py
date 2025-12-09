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
    