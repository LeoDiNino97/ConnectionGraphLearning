""" VDM.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np 


class VectorDiffusionMaps:
    """ This class implements the estimation of the connection laplacian via local PCA and vector diffusion maps allignment over a point cloud as 
    described in "Vector diffusion maps and the Connection Laplacian" (A.Singer,..., 2014)
    
    Parameters
    ----------
    X : np.ndarray
        The point cloud dataset with dimension (manifold_dim, number_of_points)
    epsilon_geometric : float
        Hyperparameter controlling the underpinning geometric graph sparsity
    epsilon_PCA : float
        Hyperparameter controlloing the PCA approximation
    gamma : float
        Hyperparameter controlling the embedded dimension estimation
    mode : str 
        Kernel function mode
    """
    def __init__ (
            self,
            X,
            epsilon_geometric,
            epsilon_PCA,
            gamma = 0.9,
            mode = 'Gaussian'
    ) -> None:
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
            Z : np.ndarray,
    ) -> np.ndarray:
        '''Kernel function applied to the euclidean distances
        
        Parameters
        ----------
        Z : np.ndarray
            Functions to be kernelezied
        
        Returns
        -------
        np.ndarray
            Kernelized functions
        '''
        supp = np.array((Z <= 1) * (0 <= Z), dtype =float)
        if self.mode == 'Gaussian':
            return np.exp(- Z ** 2) * supp
        elif self.mode == 'Epanechnikov':
            return (1 - Z ** 2) * supp
    
    def PairWiseDistances(
            self
    ) -> np.ndarray:
        '''Compute the matrix of pairwise distances between all points using the Gram trick

        Returns
        -------
        np.ndarray
            Matrix of pairwise distances
        '''
        G = self.X.T @ self.X
        S = np.diag(G)
        return np.sqrt(np.maximum(S[:,None] + S[None,:] - 2 * G, 0))
    
    def LocalPCA(
            self,
            v
    ) -> None:
        '''Compute the local representation frame of each node as the polar factor of the covariance matrix

        Parameters
        ----------
        v : int
            Label of the point at which perform local PCA

        Returns
        -------
        np.ndarray
            Left singular vectors of the neighborhood PCA
        int
            Estimated tangent space dimension
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
        '''Applying the LocalPCA routine to all the points in the dataset
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
        '''Compute a geometric graph and a connection laplacian over it via Procrustes alignment

        Returns
        -------
        np.ndarray
            Estimated connection Laplacian
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