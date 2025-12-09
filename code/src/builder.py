"""
builder.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

import numpy as np
import random
import networkx as nx
from dataclasses import dataclass

from matplotlib import pyplot as plt

from scipy.spatial import cKDTree

import gstools as gs
import pyvista as pv

from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.linalg import svd

import scipy.sparse as sp

@dataclass
class CochainSample:
    X: np.ndarray
    covariance: np.ndarray
    X_GT: np.ndarray


class ERCG():
    '''
    This class implement a ER Graph with a consistent connection graph built on top of it

    INPUT:
        V: int. The dimension of the network
        d: int. The dimension of the stalks over the nodes
        consistent: bool. A flag for the connection graph to be consistent
        special: bool. A flag for the maps to be sampled from the SO(d) group

    METHODS:
        Weights: Sample weights for each edge from Unif(0.1,3)
        Laplacian: Compute the weighted Laplacian given the weights
        RandomConnectionGraph: Builds a connection graph on top of the grid graph and deriving the related algebraic operators
        CochainsSampling: Samples from a PIAGMRF where the connection laplacian acts as the precision matrix a certain number N of samples
        Visualize: Plots the underlying graph
    '''
    
    def __init__(
            self, 
            V,
            d = 3, 
            k = 1,
            seed = 42,
            p = None,
            kron = False,
            consistent = True, 
            special = True):
        
        self.d = d
        self.V = V
        self.k = k  # Number of connected components
        self.q = self.V - self.k

        if p is None:
            p = 1.1 * np.log(self.V)/V

        self.p = p

        self.seed = seed
        
        # Fix all randomness globally
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.kron = kron
        self.consistent = consistent
        self.special = special

        def GenerateKComponentER(n, k, p):
            sizes = [n // k] * k
            for i in range(n % k):  
                sizes[i] += 1

            components = []

            for size in sizes:
                connected = False
                # Use a local variable to track tries
                local_seed = self.seed  
                while not connected:
                    component = nx.erdos_renyi_graph(size, p, seed=local_seed)
                    connected = nx.is_connected(component)
                    local_seed += 1  # increment local seed for retries but don't modify self.seed
                    
                components.append(component)

            # Relabel nodes to avoid overlaps
            G = nx.disjoint_union_all(components)
            return G

        
        self.G = GenerateKComponentER(self.V, self.k, p = self.p)

        self.A = nx.adjacency_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.W = self.Weights()
        self.L = self.Laplacian()
        self.RandomConnectionGraph()
    
    def Weights(self):
        W = np.zeros_like(self.A, dtype=float)
        for edge in self.edges:
            w = np.random.uniform(low=0.2, high=3)
            W[edge[0], edge[1]] = w
            W[edge[1], edge[0]] = w
        return W

    def Laplacian(self):
        return np.diag(np.sum(self.W, axis=0)) - self.W

    def RandomConnectionGraph(
            self
    ):
        def RandomOn(
        ):
            Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
            if self.special:
                if np.linalg.det(Q) < 0:
                    Q[0,:] *= - 1
            return Q
        
        self.nodes_frames = {
            v: None for v in range(self.V)
        }

        self.O = np.zeros((self.d*self.V, self.d*self.V))

        self.edges_map = {
            e: None for e in self.edges
        }

        # Define the bundle 
        # Define a orthonormal basis for each node:
        for v in range(self.V):
            if self.kron == True:
                self.nodes_frames[v] = np.eye(self.d)
            else:
                self.nodes_frames[v] = RandomOn()
                self.O[v * self.d : ( v + 1 ) * self.d , v * self.d : ( v + 1 ) * self.d] = np.copy(self.nodes_frames[v])
        
        # Define accordingly the On map on the respective edge
        for e in self.edges:
            if self.consistent:
                self.edges_map[e] = self.nodes_frames[e[0]].T @ self.nodes_frames[e[1]]
            else:
                self.edges_map[e] = RandomOn()

        # # Connection Adjacency matrix (block matrix)
        # self.CA = np.zeros((self.d * self.V, self.d * self.V))
        # for e in self.edges:
        #     u, v = e[0], e[1]
        #     self.CA[ u * self.d : (u + 1) * self.d, v * self.d : (v + 1) * self.d ] = self.W[e] * self.edges_map[e]
        #     self.CA[ v * self.d : (v + 1) * self.d, u * self.d : (u + 1) * self.d ] = self.W[e] * self.edges_map[e].T

        # # Connection Diagonal matrix (block matrix)
        # self.CD = np.kron(np.diag(np.sum(self.W, axis = 0)), np.eye(self.d))

        # Connection Laplacian matrix (block matrix)
        self.CL = self.O.T @ np.kron(self.L, np.eye(self.d)) @ self.O
        self.CL[np.isclose(self.CL, 0, atol=1e-6)] = 0
        
    def CochainsSampling(
            self, 
            N=1000, 
            pseudoinv=True, 
            normalized=False, 
            seed = 42,
            full=True,
            noisy=False,
            SNR = None
            ):
        
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(self.CL)

            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=N).T
            X_GT = np.copy(X)

        else:
            P = self.CL + np.eye(self.CL.shape[0])
            Sigma = np.linalg.pinv(P)

            X = np.random.randn(self.V * self.d, N)
            M = np.linalg.cholesky(Sigma)
            X = M.T @ X

        if noisy:
            assert SNR is not None, "Provide a noise variance value"
            signal_power = np.mean(X ** 2)
            SNR_linear = 10 ** (SNR / 10)

            noise_power = signal_power / SNR_linear
            X += np.random.randn(*X.shape) * np.sqrt(noise_power)
        
        if normalized:
            X = X / np.linalg.norm(X, axis=0)


        if not full:
            X = {
                v: X[v*self.d:(v + 1)*self.d, :] / np.linalg.norm(X[v*self.d:(v + 1)*self.d, :], axis=0)
                for v in range(self.V)
            }

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X if full else np.vstack([v for v in X.values()]))

        return CochainSample(
            X=X, 
            covariance=covariance, 
            X_GT = X_GT
            )
        
    def Visualize(
        self, 
        L = None,
        save_dir = None
        ):

        G = nx.Graph()

        if L is not None:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if L[i, j] != 0:
                        G.add_edge(i, j, weight=-L[i, j])  
        else:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if self.L[i, j] != 0:
                        G.add_edge(i, j, weight=-self.L[i, j])  
    
        
        # Get the edge weights from the Laplacian matrix
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        pos = nx.kamada_kawai_layout(self.G,)
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue',
            edge_color=edge_weights, width=2, edge_cmap=plt.cm.Blues, ax=ax
        )
        
        # Colorbar to show edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        plt.colorbar(sm, label='Edge Weights', ax=ax)

        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', format='pdf')

    def Topological_perturbation_sampling(
            self,
            p_tau,
            seed,
            M
    ):
        np.random.seed(seed)

        # Step 1: Identify connected components
        # -------------------------------------
        # Build adjacency matrix:
        A = -(self.L - np.diag(np.diag(self.L)))  # Laplacian -> adjacency
        A = (A > 0).astype(int)

        # Compute CCs
        visited = np.zeros(self.V, dtype=bool)
        components = []
        for i in range(self.V):
            if not visited[i]:
                stack = [i]
                comp = []
                while stack:
                    u = stack.pop()
                    if not visited[u]:
                        visited[u] = True
                        comp.append(u)
                        neighbors = np.where(A[u] > 0)[0]
                        stack.extend(neighbors)
                components.append(comp)

        # Step 2: Randomly connect different CCs
        # --------------------------------------
        L_tilde = np.copy(self.L)

        # Iterate over all pairs of *different* components
        for idx_a in range(len(components)):
            for idx_b in range(idx_a + 1, len(components)):
                compA = components[idx_a]
                compB = components[idx_b]

                # Try adding random cross-component edges
                for i in compA:
                    for j in compB:
                        if np.random.rand() < p_tau:
                            w = np.random.uniform(0.1, 0.4)

                            # Add edge to Laplacian structure
                            L_tilde[i, i] += w
                            L_tilde[j, j] += w
                            L_tilde[i, j] = -w
                            L_tilde[j, i] = -w

        # Step 3: Sampling from perturbed distribution
        # --------------------------------------------
        CL_tilde = self.O.T @ np.kron(L_tilde, np.eye(self.d)) @ self.O
        Sigma = np.linalg.pinv(CL_tilde)
        X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=M).T

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1, 1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X)

        return CochainSample(
            X=X,
            covariance=covariance,
            X_GT=X
        ), L_tilde


    def Geometric_perturbation_sampling(
            self,
            p_iota,
            seed,
            M
    ):
        np.random.seed(seed)

        def RandomOn():
            Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
            if self.special and np.linalg.det(Q) < 0:
                Q[0,:] *= -1
            return Q

        # Perturbation of transport maps
        L_tilde = np.copy(self.CL)
        for e in self.edges:
            if np.random.rand() < p_iota:
                u = e[0]
                v = e[1]
                R = RandomOn()

                L_uv = L_tilde[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] 
                L_vu = L_tilde[v * self.d : ( v + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] 

                L_tilde[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] = R @ L_uv
                L_tilde[v * self.d : ( v + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] = L_vu @ R.T

        # Sampling from perturbed distribution
        Sigma = np.linalg.pinv(L_tilde)
        X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=M).T   

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X)

        return CochainSample(
            X=X, 
            covariance=covariance, 
            X_GT = X
        )

class RBFCG():

    def __init__(
            self, 
            V,
            d = 3, 
            seed = 42,
            kron = False,
            consistent = True, 
            special = True):
        
        self.d = d
        self.V = V

        self.seed = seed
        
        # Fix all randomness globally
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.kron = kron
        self.consistent = consistent
        self.special = special

        def GenerateGeometricGaussianWeighted(N, sigma, dim=3, seed=None):
            """
            Generate a weighted geometric graph in the unit cube [0,1]^dim.
            
            Edge weight between nodes i and j is:
                w_ij = exp(-||x_i - x_j||^2 / (2 * sigma^2))
            """
            rng = np.random.default_rng(seed)

            # 1. Sample coordinates in [0,1]^dim
            X = rng.random((N, dim))

            # 2. Pairwise squared distances (vectorized)
            diff = X[:, None, :] - X[None, :, :]
            dist2 = np.sum(diff**2, axis=2)

            # 3. Gaussian kernel weights
            W = np.exp(-dist2 / (2 * sigma**2))
            # W[W > np.quantile(W, 0.4)] = 0
            W[W < 0.75] = 0

            # 4. Remove self-weights (optional, set diag to 0)
            np.fill_diagonal(W, 0.0)

            # 5. Create weighted graph
            G = nx.from_numpy_array(W)

            return G

        self.G = GenerateGeometricGaussianWeighted(self.V, sigma =0.5,)

        self.L = nx.laplacian_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.RandomConnectionGraph()

    def RandomConnectionGraph(
            self
    ):
        def RandomOn(
        ):
            Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
            if self.special:
                if np.linalg.det(Q) < 0:
                    Q[0,:] *= - 1
            return Q
        
        self.nodes_frames = {
            v: None for v in range(self.V)
        }

        self.O = np.zeros((self.d*self.V, self.d*self.V))

        self.edges_map = {
            e: None for e in self.edges
        }

        # Define the bundle 
        # Define a orthonormal basis for each node:
        for v in range(self.V):
            if self.kron == True:
                self.nodes_frames[v] = np.eye(self.d)
            else:
                self.nodes_frames[v] = RandomOn()
                self.O[v * self.d : ( v + 1 ) * self.d , v * self.d : ( v + 1 ) * self.d] = np.copy(self.nodes_frames[v])
        
        # Define accordingly the On map on the respective edge
        for e in self.edges:
            if self.consistent:
                self.edges_map[e] = self.nodes_frames[e[0]].T @ self.nodes_frames[e[1]]
            else:
                self.edges_map[e] = RandomOn()

        # # Connection Adjacency matrix (block matrix)
        # self.CA = np.zeros((self.d * self.V, self.d * self.V))
        # for e in self.edges:
        #     u, v = e[0], e[1]
        #     self.CA[ u * self.d : (u + 1) * self.d, v * self.d : (v + 1) * self.d ] = self.W[e] * self.edges_map[e]
        #     self.CA[ v * self.d : (v + 1) * self.d, u * self.d : (u + 1) * self.d ] = self.W[e] * self.edges_map[e].T

        # # Connection Diagonal matrix (block matrix)
        # self.CD = np.kron(np.diag(np.sum(self.W, axis = 0)), np.eye(self.d))

        # Connection Laplacian matrix (block matrix)
        self.CL = self.O.T @ np.kron(self.L, np.eye(self.d)) @ self.O
        self.CL[np.isclose(self.CL, 0, atol=1e-6)] = 0
        
    def CochainsSampling(
            self, 
            N=1000, 
            pseudoinv=True, 
            normalized=False, 
            seed = 42,
            full=True,
            noisy=False,
            SNR = None
            ):
        
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(self.CL)

            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=N).T
            X_GT = np.copy(X)

        else:
            P = self.CL + np.eye(self.CL.shape[0])
            Sigma = np.linalg.pinv(P)

            X = np.random.randn(self.V * self.d, N)
            M = np.linalg.cholesky(Sigma)
            X = M.T @ X

        if noisy:
            assert SNR is not None, "Provide a noise variance value"
            signal_power = np.mean(X ** 2)
            SNR_linear = 10 ** (SNR / 10)

            noise_power = signal_power / SNR_linear
            X += np.random.randn(*X.shape) * np.sqrt(noise_power)
        
        if normalized:
            X = X / np.linalg.norm(X, axis=0)


        if not full:
            X = {
                v: X[v*self.d:(v + 1)*self.d, :] / np.linalg.norm(X[v*self.d:(v + 1)*self.d, :], axis=0)
                for v in range(self.V)
            }

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X if full else np.vstack([v for v in X.values()]))

        return CochainSample(
            X=X, 
            covariance=covariance, 
            X_GT = X_GT
            )
        
    def Visualize(
        self, 
        L = None,
        save_dir = None
        ):

        G = nx.Graph()

        if L is not None:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if L[i, j] != 0:
                        G.add_edge(i, j, weight=-L[i, j])  
        else:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if self.L[i, j] != 0:
                        G.add_edge(i, j, weight=-self.L[i, j])  
    
        
        # Get the edge weights from the Laplacian matrix
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        pos = nx.kamada_kawai_layout(self.G,)
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue',
            edge_color=edge_weights, width=2, edge_cmap=plt.cm.Blues, ax=ax
        )
        
        # Colorbar to show edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        plt.colorbar(sm, label='Edge Weights', ax=ax)

        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', format='pdf')

class SBMCG():

    def __init__(
            self, 
            V,
            d = 3, 
            seed = 42,
            k = 3,
            p_k = None,
            p_in = None,
            p_out = None,
            kron = False,
            consistent = True, 
            special = True):
        
        self.d = d
        self.V = V

        self.seed = seed
        
        # Stochastic block model parameters
        self.k = k                             # Number of communities
        self.p_k = p_k / np.sum(p_k)           # Communities distribution assignment
        self.p_in = p_in                       # Intra cluster wiring probability
        self.p_out = p_out                     # Inter cluster wiring probability

        self.A = np.zeros((V,V))

        # Fix all randomness globally
        np.random.seed(self.seed)
        random.seed(self.seed)

        self.kron = kron
        self.consistent = consistent
        self.special = special
        
        # Build graph
        self.graph_building()

        # Convert to networkx graph
        self.G = nx.from_numpy_array(self.A)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.RandomConnectionGraph()

    # --- Proper class methods ---

    def block_assignment(self):
        self.B = {node: np.random.choice(self.k, p=self.p_k)
                  for node in range(self.V)}

    def block_transition(self):
        self.C = np.ones((self.k, self.k)) * self.p_out
        self.C[range(self.k), range(self.k)] = self.p_in

    def edge_generation(self):
        for i in range(self.V):
            for j in range(i + 1, self.V):
                U = self.B[i]
                V = self.B[j]
                prob = self.C[U, V]
                edge = np.random.rand() < prob
                self.A[i, j] = edge
                self.A[j, i] = edge

    def graph_building(self):
        self.block_assignment()
        self.block_transition()
        self.edge_generation()

    def RandomConnectionGraph(
            self
    ):
        def RandomOn(
        ):
            Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
            if self.special:
                if np.linalg.det(Q) < 0:
                    Q[0,:] *= - 1
            return Q
        
        self.nodes_frames = {
            v: None for v in range(self.V)
        }

        self.O = np.zeros((self.d*self.V, self.d*self.V))

        self.edges_map = {
            e: None for e in self.edges
        }

        # Define the bundle 
        # Define a orthonormal basis for each node:
        for v in range(self.V):
            if self.kron == True:
                self.nodes_frames[v] = np.eye(self.d)
            else:
                self.nodes_frames[v] = RandomOn()
                self.O[v * self.d : ( v + 1 ) * self.d , v * self.d : ( v + 1 ) * self.d] = np.copy(self.nodes_frames[v])
        
        # Define accordingly the On map on the respective edge
        for e in self.edges:
            if self.consistent:
                self.edges_map[e] = self.nodes_frames[e[0]].T @ self.nodes_frames[e[1]]
            else:
                self.edges_map[e] = RandomOn()

        # # Connection Adjacency matrix (block matrix)
        # self.CA = np.zeros((self.d * self.V, self.d * self.V))
        # for e in self.edges:
        #     u, v = e[0], e[1]
        #     self.CA[ u * self.d : (u + 1) * self.d, v * self.d : (v + 1) * self.d ] = self.W[e] * self.edges_map[e]
        #     self.CA[ v * self.d : (v + 1) * self.d, u * self.d : (u + 1) * self.d ] = self.W[e] * self.edges_map[e].T

        # # Connection Diagonal matrix (block matrix)
        # self.CD = np.kron(np.diag(np.sum(self.W, axis = 0)), np.eye(self.d))

        # Connection Laplacian matrix (block matrix)
        self.CL = self.O.T @ np.kron(self.L, np.eye(self.d)) @ self.O
        self.CL[np.isclose(self.CL, 0, atol=1e-6)] = 0
        
    def CochainsSampling(
            self, 
            N=1000, 
            pseudoinv=True, 
            normalized=False, 
            seed = 42,
            full=True,
            noisy=False,
            SNR = None
            ):
        
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(self.CL)

            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=N).T
            X_GT = np.copy(X)

        else:
            P = self.CL + np.eye(self.CL.shape[0])
            Sigma = np.linalg.pinv(P)

            X = np.random.randn(self.V * self.d, N)
            M = np.linalg.cholesky(Sigma)
            X = M.T @ X

        if noisy:
            assert SNR is not None, "Provide a noise variance value"
            signal_power = np.mean(X ** 2)
            SNR_linear = 10 ** (SNR / 10)

            noise_power = signal_power / SNR_linear
            X += np.random.randn(*X.shape) * np.sqrt(noise_power)
        
        if normalized:
            X = X / np.linalg.norm(X, axis=0)


        if not full:
            X = {
                v: X[v*self.d:(v + 1)*self.d, :] / np.linalg.norm(X[v*self.d:(v + 1)*self.d, :], axis=0)
                for v in range(self.V)
            }

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X if full else np.vstack([v for v in X.values()]))

        return CochainSample(
            X=X, 
            covariance=covariance, 
            X_GT = X_GT
            )
        
    def Visualize(
        self, 
        L = None,
        save_dir = None
        ):

        G = nx.Graph()

        if L is not None:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if L[i, j] != 0:
                        G.add_edge(i, j, weight=-L[i, j])  
        else:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if self.L[i, j] != 0:
                        G.add_edge(i, j, weight=-self.L[i, j])  
    
        
        # Get the edge weights from the Laplacian matrix
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]

        pos = nx.spring_layout(self.G,)
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(
            G, pos, with_labels=True, node_color='lightblue',
            edge_color='blue', width=2, edge_cmap=plt.cm.Blues, ax=ax
        )
        
        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', format='pdf')

class GridCG():
    '''
    Implements a 2D grid graph with a consistent connection graph built on top of it.

    INPUT:
        side: int. The number of nodes along one side of the grid (total nodes = side^2)
        d: int. The dimension of the stalks over the nodes
        consistent: bool. Whether the connection graph is consistent
        special: bool. Whether the maps are sampled from SO(d)

    METHODS:
        Weights: Sample weights for each edge from Unif(0.1,3)
        Laplacian: Compute the weighted Laplacian
        RandomConnectionGraph: Builds a connection graph on top of the grid and derives operators
        CochainsSampling: Samples from a PIAGMRF where the connection Laplacian acts as the precision
        Visualize: Plots the underlying grid graph
    '''
    
    def __init__(
            self, 
            side,
            d=3, 
            seed=42,
            kron=False,
            consistent=True, 
            special=True):
        
        self.side = side
        self.V = side**2
        self.d = d
        self.k = 1  # only one connected component for grid
        self.q = self.V - self.k

        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.kron = kron
        self.consistent = consistent
        self.special = special

        # === Build the 2D grid graph ===
        self.G = nx.grid_2d_graph(side, side)
        # Relabel (i, j) -> single integer node
        mapping = {node: idx for idx, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)

        self.A = nx.adjacency_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        # === Assign weights and Laplacian ===
        self.W = self.Weights()
        self.L = self.Laplacian()

        # === Build connection graph ===
        self.RandomConnectionGraph()
    
    def Weights(self):
        W = np.zeros_like(self.A, dtype=float)
        for edge in self.edges:
            w = np.random.uniform(low=0.2, high=3)
            W[edge[0], edge[1]] = w
            W[edge[1], edge[0]] = w
        return W

    def Laplacian(self):
        return np.diag(np.sum(self.W, axis=0)) - self.W

    def RandomConnectionGraph(self):
        def RandomOn():
            Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
            if self.special and np.linalg.det(Q) < 0:
                Q[0,:] *= -1
            return Q
        
        self.nodes_frames = {v: None for v in range(self.V)}
        self.O = np.zeros((self.d*self.V, self.d*self.V))
        self.edges_map = {e: None for e in self.edges}

        # Define local orthonormal bases for each node
        for v in range(self.V):
            if self.kron:
                self.nodes_frames[v] = np.eye(self.d)
            else:
                self.nodes_frames[v] = RandomOn()
                self.O[v*self.d:(v+1)*self.d, v*self.d:(v+1)*self.d] = self.nodes_frames[v]
        
        # Define connection maps on edges
        for e in self.edges:
            if self.consistent:
                self.edges_map[e] = self.nodes_frames[e[0]].T @ self.nodes_frames[e[1]]
            else:
                self.edges_map[e] = RandomOn()

        # Connection Laplacian (block matrix)
        self.CL = self.O.T @ np.kron(self.L, np.eye(self.d)) @ self.O
        self.CL[np.isclose(self.CL, 0, atol=1e-6)] = 0
        
    def CochainsSampling(
            self, 
            N=1000, 
            pseudoinv=True, 
            normalized=False, 
            seed=42,
            full=True,
            noisy=False,
            SNR=None):
        
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(self.CL)
            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=N).T
            X_GT = np.copy(X)
        else:
            P = self.CL + np.eye(self.CL.shape[0])
            Sigma = np.linalg.pinv(P)
            X = np.random.randn(self.V * self.d, N)
            M = np.linalg.cholesky(Sigma)
            X = M.T @ X
            X_GT = np.copy(X)

        if noisy:
            assert SNR is not None, "Provide a noise variance value"
            signal_power = np.mean(X ** 2)
            SNR_linear = 10 ** (SNR / 10)
            noise_power = signal_power / SNR_linear
            X += np.random.randn(*X.shape) * np.sqrt(noise_power)
        
        if normalized:
            X = X / np.linalg.norm(X, axis=0)

        if not full:
            X = {
                v: X[v*self.d:(v+1)*self.d, :] / np.linalg.norm(X[v*self.d:(v+1)*self.d, :], axis=0)
                for v in range(self.V)
            }

        def SampleCovariance(X):
            X_mean = np.mean(X, axis=1)
            X_centered = X - X_mean.reshape(-1,1)
            S = (X_centered @ X_centered.T) / (X_centered.shape[1] - 1)
            return S

        covariance = SampleCovariance(X if full else np.vstack([v for v in X.values()]))

        return CochainSample(
            X=X, 
            covariance=covariance, 
            X_GT = X_GT
            )
        
    def Visualize(self, L=None, save_dir=None):
        G = nx.Graph()
        if L is not None:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if L[i, j] != 0:
                        G.add_edge(i, j, weight=-L[i, j])  
        else:
            for i in range(self.V):
                for j in range(i + 1, self.V):
                    if self.L[i, j] != 0:
                        G.add_edge(i, j, weight=-self.L[i, j])  
    
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        fig, ax = plt.subplots(figsize=(8, 8))
        nx.draw(
            G,  pos = {i * self.side + j: (j, -i) for i in range(self.side) for j in range(self.side)}, with_labels=True, node_color='lightblue',
            edge_color=edge_weights, width=2, edge_cmap=plt.cm.Blues, ax=ax
        )
        
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, 
                                norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        plt.colorbar(sm, label='Edge Weights', ax=ax)

        if save_dir is not None:
            plt.savefig(save_dir, dpi=300, bbox_inches='tight', format='pdf')
     
class FibonacciSphereGraph:
    """
    Generates a Fibonacci-lattice sphere and constructs a k-NN graph over it with graph alignment and VDM approximation
    """
    
    def __init__(
            self, 
            epsilon_geometric = 1,
            V = 200, 
            k_neighbors = 6,
            kernel_mode = 'Gaussian'):
        
        self.epsilon_geometric = epsilon_geometric
        self.V = V
        self.k_neighbors = k_neighbors

        self.kernel_mode = kernel_mode

        self.d = 3
        self.d_hat = 2

        self._generate_points()
        self._build_knn_graph()
        self._graph_alingment()
        self._connection_laplacian()
        self._spectral_synchronization(op = 'Normalized')
        self._spectral_synchronization(op = 'Unnormalized')
    
    def _generate_points(self):
        '''
        Generate a Fibonacci lattice on the unit sphere
        '''
        
        indices = np.arange(0, self.V, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices/self.V)
        theta = np.pi * (1 + 5**0.5) * indices
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        self.points = np.vstack((x, y, z)).T

    def _build_knn_graph(self):
        '''
        Construct a k-NN graph from the Fibonacci lattice
        '''
        tree = cKDTree(self.points)
        edge_set = set()
        edges = []

        for i, point in enumerate(self.points):
            _, neighbors = tree.query(point, k=self.k_neighbors + 1)
            for j in neighbors[1:]:  # skip self
                if (i, j) not in edge_set and (j, i) not in edge_set:
                    edges.append((i, j))
                    edge_set.add((i, j))

        n_edges = len(edges)
        B = np.zeros((self.V, n_edges), dtype=float)

        for edge_id, (i, j) in enumerate(edges):
            B[i, edge_id] = 1
            B[j, edge_id] = -1

        self.edges = edges
        self.B = B
        self.L = B @ B.T
        self.A = self.L - np.diag(np.diag(self.L))
    
    def _graph_alingment(self):
        '''
        Implement the graph Local PCA as in Barbero F., "Sheaf Neural Networks with Connection Laplacians"
        '''
        self.local_bases = {v: None for v in range(self.V)}

        for v in range(self.V):
            neighbors_index = np.where(self.A[v,:] != 0)[0]
            X_v = self.points[neighbors_index, ]
            U, _, _ = np.linalg.svd(X_v.T)
            self.local_bases[v] = U[:, 0 : 2]   

    def _connection_laplacian(
            self
    ):
        '''
        Compute a geometric graph and a connection laplacian over it via Procrustes alignment
        '''

        # Compute the Procrustes transformation and some accessory stuff
        self.maps = {}
        self.degree_matrices = {}
        self.ndegree_matrices = {}

        self.D = np.kron(np.diag(np.diag(self.L)), np.eye(self.d_hat))
        self.DN = np.sqrt(np.linalg.inv(self.D))
        self.O = np.zeros((self.V * self.d_hat, self.V * self.d_hat))

        for i in range(self.V):
            # Retrieve node SO(n) basis
            # U, _, Vt = np.linalg.svd(self.local_bases[i].T @ self.local_bases[i])
            # M = U @ Vt
            # if np.linalg.det(M) < 0:
            #     M[0,:] *= - 1

            # self.O[ i * self.d_hat : ( i + 1 ) * self.d_hat, i * self.d_hat : ( i + 1 ) * self.d_hat] = M

            for j in range(i + 1, self.V):
                if self.L[i, j] != 0:

                    O_ij_tilde = self.local_bases[i].T @ self.local_bases[j]
                    U, _, Vt = np.linalg.svd(O_ij_tilde)
                    O_ij = U @ Vt

                    self.maps[(i,j)] = O_ij

        # Computing the normalized connection Laplacian and the unnormalized connection Laplacian 
        self.S = np.zeros((self.V * self.d_hat, self.V * self.d_hat))
    
        for edge in self.maps.keys():
            i = edge[0]
            j = edge[1]

            self.S[i * self.d_hat : ( i + 1 ) * self.d_hat, j * self.d_hat : ( j + 1 ) * self.d_hat] = self.maps[edge]
            self.S[j * self.d_hat : ( j + 1 ) * self.d_hat, i * self.d_hat : ( i + 1 ) * self.d_hat] = self.maps[edge].T

        self.CLN = self.DN @ ( self.S - self.D ) @ self.DN                                  # Normalized connection Laplacian
        self.CLUN = self.S - self.D                                                         # Unnormalized connection Laplacian
        # self.CLUN_FLAT = - ( self.O.T @ np.kron(self.L, np.eye(self.d_hat)) @ self.O )      # Flat bundle connection Laplacian

        self.laplacians = {
            'Normalized': self.CLN,
            'Unnormalized': self.CLUN,
            # 'FlatBundle': self.CLUN_FLAT
        }

    def _proj_to_O(self, M):
        """Project a d_hat x d_hat matrix to O(d_hat) via SVD; optional force SO(d_hat)."""
        U, _, Vt = svd(M, full_matrices=False)
        R = U @ Vt
        # Optional: force rotation (SO) instead of reflection + rotation:
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        return R

    def _spectral_synchronization(self, op='Unnormalized', k=None):
        """
        Run spectral synchronization on a chosen connection Laplacian (Normalized or Unnormalized).
        Returns:
        g : list of n (d_hat x d_hat) orthogonal node frames
        F : dict (i,j) -> d_hat x d_hat flat transports = g_i^T g_j (only for edges present in self.maps)
        Also stores assembled flat Laplacian under self.laplacians['FlatFromSync_{op}'].
        """
        if op not in self.laplacians:
            raise ValueError(f"op must be one of {list(self.laplacians.keys())}")

        Lc = self.laplacians[op]
        n = self.V
        d = self.d_hat
        N = n * d

        # Number of eigenvectors to extract: default = d (the fiber dimension)
        if k is None:
            k = d

        # We expect Lc to be symmetric; convert to sparse if dense large
        try:
            # Use eigsh for symmetric/hermitian
            vals, vecs = eigsh(sp.csr_matrix(Lc), k=k, which='SM')
        except Exception:
            # Fallback to dense eigh if eigsh fails (small graphs)
            vals_all, vecs_all = eigh(Lc)
            vecs = vecs_all[:, :k]

        U = vecs.real  # N x d

        # Build orthogonal frames g_i by SVD projection of node blocks
        g = []
        for i in range(n):
            block = U[i*d:(i+1)*d, :]  # d x d
            # If block is close to rank-deficient, SVD projection still works
            Ri = self._proj_to_O(block)
            g.append(Ri)

        # Build flat transports F_ij = g_i^T g_j only for edges present in maps
        F = {}
        for (i, j) in self.maps.keys():
            F[(i, j)] = g[i].T @ g[j]
            F[(j, i)] = F[(i, j)].T

        blocks = []
        for i in range(n):
            blocks.append(g[i])
        O_sync = sp.block_diag(blocks, format='csr')   

        L_graph = sp.csr_matrix(self.L)  # V x V
        I_d = sp.eye(d, format='csr')
        Kron = sp.kron(L_graph, I_d, format='csr')    

        CL_flat_sync = - (O_sync.T @ Kron @ O_sync)   

        # Store into laplacians dictionary 
        key = f'FlatFromSync_{op}'
        self.laplacians[key] = CL_flat_sync.toarray() if sp.issparse(CL_flat_sync) else CL_flat_sync

    def _tangent_bundle_signal(self, X):
        '''
        Project a vector field over the discretized tangent bundle
        '''
        
        f = np.zeros((2 * self.V, X.shape[1]))
        for v in range(self.V):
            f[v * 2 : ( v + 1 ) * 2] = self.local_bases[v].T @ X[v * 3 : ( v + 1 ) * 3]
        
        return f
    
    def _cochains_sampling(
            self, 
            N=1000, 
            pseudoinv=True, 
            normalized=False, 
            seed = 42,
            full=True,
            op = 'FlatFromSync_Unnormalized'
            ):
        
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(- self.laplacians[op])
            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d_hat), cov=Sigma, size=N).T

        else:
            P = - self.laplacians[op] + np.eye(self.laplacians[op].shape[0])
            Sigma = np.linalg.pinv(P)

            X = np.random.randn(self.V * self.d_hat, N)
            M = np.linalg.cholesky(Sigma)
            X = M.T @ X

        if normalized:
            X = X / np.linalg.norm(X, axis=0)

        if not full:
            X = {
                v: X[v*self.d_hat:(v + 1)*self.d_hat, :] / np.linalg.norm(X[v*self.d_hat:(v + 1)*self.d_hat, :], axis=0)
                for v in range(self.V)
            }

        return X
    