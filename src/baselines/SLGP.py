"""SLGP.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np 
from sklearn.model_selection import KFold
from scipy.fft import dct


class SmoothSheafDiffusion:
    """Implementation of a sheaf learning algorithm from
    'Learning Sheaf Laplacian optimizing restriction maps' (L.Di Nino, et al.,  2024)
    based on local compression, Procrustes alignment and hierarchical edge sampling

    Parameters
    ----------
    X : int
        Observed 0-cochains in the shape (V * d, n_samples)
    V : int
        Number of nodes in the network
    d : int
        Stalk dimension
    n_edges : int
        Number of edges
    k_folds : int
        Number of folds for the validation procedure
    """
    def __init__(
        self,
        X : np.ndarray,
        V : int,
        d : int,
        n_edges : int,
        k_folds : int = 5
    ) -> None:

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
        self.best_alpha = self.CrossValidation(k_folds)
        self.LocalSparsifying(self.best_alpha)
        self.Alignment()
    
    def CrossValidation(
        self, 
        k_folds : int = 5
    ) -> float:
        """The method performs the cross validation for the hyperparameter regulating the local sparsification
        
        Parameters
        ----------
        k_folds : int
            Number of folds for the validation procedure

        Returns
        -------
        float
            Best alpha parameter
        """
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

    def LocalSparsifying(
        self, 
        alpha : float, 
        X : np.ndarray = None
    ):
        """Performs local sparsification on the given basis (DCT)       

        Parameters
        ----------
        alpha : float
            Parameter regulating the sparsification
        X : np.ndarray
            Signals to be processed
        """
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

    def Alignment(
        self
    ):
        """ Performs pairwise alignment based on Procrustes solution
        """
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

    def LaplacianBuilder(
        self
    ) -> np.ndarray:
        """ Build a Sheaf Laplacian according to the prior on the number of edges and the post-alingment distances

        Returns
        -------
        np.ndarray
            The estimated sheaf Laplacian
        """

        edges = list(sorted(self.dists.items(), key=lambda x:x[1]))[ : self.n_edges]
        L = np.zeros((self.d * self.V, self.d * self.V))

        for edge in edges: 
            u = edge[0][0]
            v = edge[0][1] 

            L[u * self.d : ( u + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] += np.eye(self.d)
            L[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] = - self.maps[(u,v)][u]

        return L 
    