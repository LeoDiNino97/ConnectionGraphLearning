""" builder.py

Module for "Structured Learning of Consistent Connection Laplacians with Spectral Constraints", Di Nino L., D'Acunto G., et al., 2025
@Author: Leonardo Di Nino
Date: 2025-04
"""

from dataclasses import dataclass

import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

@dataclass
class CochainSample:
    """ Wrapper for a cochain sample
    """
    X: np.ndarray
    covariance: np.ndarray
    X_GT: np.ndarray

class CG:
    """ This is the super class for connection graphs

    Parameters
    ----------

    V : int
        Number of nodes in the network
    d : int
        Fiber (stalk) dimension
    seed : int
        Global seed
    kron : bool
        Flag for trivial bundle
    consistent : bool
        Flag for consistency
    special : bool
        Flag for special orthogonal group
    """
    def __init__(
        self,
        V : int,
        d : int,
        seed : int = 42,
        kron : bool = False,
        consistent : bool = True,
        special : bool = True
    ) -> None:
        
        # System dimensions
        self.V = V
        self.d = d

        # Flags and seeds
        self.seed = seed
        np.random.seed(self.seed)

        # CG properties
        self.kron = kron
        self.consistent = consistent
        self.special = special

        # Placeholders for subclasses
        self.L = None
        self.edges = None

    def random_On(
        self
    ) -> np.ndarray:
        """ Generates a random d-dimensional orthogonal matrix

        Returns
        -------
        np.ndarray
            Matrix sampled from O(d) 
        """
        Q, _ = np.linalg.qr(np.random.randn(self.d, self.d))
        if self.special:
            if np.linalg.det(Q) < 0:
                Q[0,:] *= - 1
        return Q
    
    def random_connection_graph(
        self
    ):
        """ Generates a random d-dimensional connection over the graph
        """

        assert self.edges is not None, "Edges must be defined"
        assert self.L is not None, "Laplacian must be defined"
    
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
                self.nodes_frames[v] = self.random_On()
                self.O[v * self.d : ( v + 1 ) * self.d , v * self.d : ( v + 1 ) * self.d] = np.copy(self.nodes_frames[v])
        
        # Define accordingly the On map on the respective edge
        for e in self.edges:
            if self.consistent:
                self.edges_map[e] = self.nodes_frames[e[0]].T @ self.nodes_frames[e[1]]
            else:
                self.edges_map[e] = self.random_On()

        # Connection Laplacian matrix (block matrix)
        self.CL = self.O.T @ np.kron(self.L, np.eye(self.d)) @ self.O
        self.CL[np.isclose(self.CL, 0, atol=1e-6)] = 0
        
    def cochains_sampling(
        self, 
        N : int = 1000, 
        CL : np.ndarray = None,
        pseudoinv : bool = True, 
        normalized : bool = False, 
        seed : int = 42,
        full : bool = True,
        noisy : bool = False,
        SNR : float = None
    ) -> CochainSample:
        """ Method to sample 0-cochains from N(0, pinv(L) + sigma^2I)

        Parameters
        ----------
        N : int
            Number of signals
        CL : np.ndarray
            Placeholder to support different laplacian for perturbed sampling
        pseudoinv : bool
            Flag to whether use the pinv of the laplacian
        normalized : bool 
            Flag to whether normalize signals
        full : bool
            Flag to whether signals should be unpacked in local measurementes
        noise : bool
            Flag for AWGN
        SNR : float
            SNR for noisy signals

        Returns
        -------
        CochainSample
            Generated 0-cochains
        """

        CL_ = CL if CL is not None else self.CL 
        np.random.seed(seed)
        if pseudoinv:
            Sigma = np.linalg.pinv(CL_)
            X = np.random.multivariate_normal(mean=np.zeros(self.V * self.d), cov=Sigma, size=N).T
            X_GT = np.copy(X)

        else:
            P = self.CL + np.eye(CL_.shape[0])
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
        
    def visualize(
        self, 
        L : np.ndarray = None,
        save_dir : str = None
    ) -> None:
        """ Visualize the network with the given structure or the estimated one
        Parameters
        ----------
        L : np.ndarray
            Estimated Laplacian to visualize inferred networks
        save_dir : str
            Save directory for visualization     
        """

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

    def topological_perturbation_sampling(
        self,
        p_tau : float,
        M : int,
        seed : int = 42,
        pseudoinv : bool = True, 
        normalized : bool = False, 
        full : bool = True,
        noisy : bool = False,
        SNR : float = None
    ) -> CochainSample:
        """ Sampling from a topologically perturbed version of the CG

        Parameters
        ----------

        p_tau : float
            Probability of connecting 
        M : int
            Number of signals to be generated
        seed : int
            Seed for reproducibility
        pseudoinv : bool
            Flag to whether use the pinv of the laplacian
        normalized : bool 
            Flag to whether normalize signals
        full : bool
            Flag to whether signals should be unpacked in local measurementes
        noise : bool
            Flag for AWGN
        SNR : float
            SNR for noisy signals

        Returns
        -------
        
        CochainsSample
            Sampled 0-cochains
        """
        np.random.seed(seed)

        # Identify connected components CCs
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

        # Randomly connect different CCs
        L_tilde = np.copy(self.L)

        # Iterate over all pairs of different components
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

        # Sampling from perturbed distribution
        CL_tilde = self.O.T @ np.kron(L_tilde, np.eye(self.d)) @ self.O
        X = self.cochains_sampling(N = M, CL = CL_tilde, pseudoinv= pseudoinv, normalized = normalized, seed = seed, full = full, noisy = noisy, SNR = SNR)

        return X, L_tilde
    
    def geometric_perturbation_sampling(
        self,
        p_iota : float,
        M : int,
        seed : int = 42,
        pseudoinv : bool = True, 
        normalized : bool = False, 
        full : bool = True,
        noisy : bool = False,
        SNR : float = None
    ) -> CochainSample:
        """ Sampling from a geometrically perturbed version of the CG

        Parameters
        ----------

        p_iota : float
            Probability of perturbing the connection over an edge
        M : int
            Number of signals to be generated
        seed : int
            Seed for reproducibility
        pseudoinv : bool
            Flag to whether use the pinv of the laplacian
        normalized : bool 
            Flag to whether normalize signals
        full : bool
            Flag to whether signals should be unpacked in local measurementes
        noise : bool
            Flag for AWGN
        SNR : float
            SNR for noisy signals
        Returns
        -------
        CochainSample
            Sampled 0-cochains
        """
        np.random.seed(seed)

        # Perturbation of transport maps
        CL_tilde = np.copy(self.CL)
        for e in self.edges:
            if np.random.rand() < p_iota:
                u = e[0]
                v = e[1]
                R = self.random_On()

                L_uv = CL_tilde[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] 
                L_vu = CL_tilde[v * self.d : ( v + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] 

                CL_tilde[u * self.d : ( u + 1 ) * self.d, v * self.d : ( v + 1 ) * self.d] = R @ L_uv
                CL_tilde[v * self.d : ( v + 1 ) * self.d, u * self.d : ( u + 1 ) * self.d] = L_vu @ R.T
                
        # Sampling from perturbed distribution
        X = self.cochains_sampling(N = M, CL = CL_tilde, pseudoinv= pseudoinv, normalized = normalized, seed = seed, full = full, noisy = noisy, SNR = SNR)

        return X, CL_tilde
    
class ERCG(CG):
    """ This class implement a ER Connection Graph with a consistent connection graph built on top of it enforcing a certain number of connected components

    Parameters
    ----------
    V : int 
        Number of nodes in the network
    d : int 
        Fiber (stalk) dimension
    k : int
        Number of connected components
    seed : float
        Seed for randomness
    p : float
        Probability of connection 
    kron : bool
        Flag for trivial bundle
    consistent : bool
        Flag for consistency
    special : bool
        Flag for special orthogonal group
    """
    
    def __init__(
        self, 
        V : int,
        d : int, 
        k : int,
        w_LB : float = 0.2,
        w_UB : float = 3,
        seed = 42,
        p = None,
        kron = False,
        consistent = True, 
        special = True
    ) -> None:
        super().__init__(V, d, seed, kron, consistent, special)

        self.k = k  # Number of connected components
        self.q = self.V - self.k
        
        # Bounds on the weight distribution
        self.w_LB = w_LB 
        self.w_UB = w_UB

        if p is None:
            p = 1.1 * np.log(self.V)/V

        self.p = p

        def generate_Kcomponent_ER():
            """ Helper to generate a ER graph with a given number of connected components   

            Returns
            -------
            nx.Graph
                Istantiation from networkx graph
            """
            sizes = [self.V // k] * self.k
            for i in range(self.V % self.k):  
                sizes[i] += 1

            components = []

            for size in sizes:
                connected = False
                # Use a local variable to track tries
                local_seed = self.seed  
                while not connected:
                    component = nx.erdos_renyi_graph(size, self.p, seed=local_seed)
                    connected = nx.is_connected(component)
                    local_seed += 1  # increment local seed for retries but don't modify self.seed
                    
                components.append(component)

            # Relabel nodes to avoid overlaps
            G = nx.disjoint_union_all(components)
            return G

        self.G = generate_Kcomponent_ER()

        self.A = nx.adjacency_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.W = self.weights()
        self.L = self.laplacian()
        self.random_connection_graph()
    
    def weights(
        self
    ) -> np.ndarray:
        """ Samples edge weights matrix
        
        Returns
        -------
        np.ndarray
            Weighted Adjacency matrix
        """
        W = np.zeros_like(self.A, dtype=float)
        for edge in self.edges:
            w = np.random.uniform(low=0.2, high=3)
            W[edge[0], edge[1]] = w
            W[edge[1], edge[0]] = w
        return W

    def laplacian(
        self
    ) -> np.ndarray:
        """ Returns a valid weighted laplacian 
        
        Returns
        -------
        np.ndarray
            Combinatorial laplacian
        """
        return np.diag(np.sum(self.W, axis=0)) - self.W

class RBFCG(CG):
    """ This class implements a Random Geometric Graph with a consistent connection graph built on top of it

    Parameters
    ----------
    V : int 
        Number of nodes in the network
    d : int 
        Fiber (stalk) dimension
    sigma : float
        Parameter for the bandwidth of the Gaussian kernel
    cutoff : float
        Threshold to cut edges
    seed : float
        Seed for randomness
    kron : bool
        Flag for trivial bundle
    consistent : bool
        Flag for consistency
    special : bool
        Flag for special orthogonal group
    """
    def __init__(
        self, 
        V : int,
        d : int, 
        sigma : float = 0.5,
        cutoff : float = 0.75,
        seed : int = 42,
        kron : bool = False,
        consistent : bool = True, 
        special : bool = True
    ) -> None:
        super().__init__(V, d, seed, kron, consistent, special)

        def generate_geometric_gaussian_weighted():
            """ Generate a RBF graph sampling V points in [0,1]^d

            Returns
            -------
            nx.Graph
                Istantiation from networkx Graph object
            """

            # Sample coordinates in [0,1]^(d+1)
            X = np.random.rand(self.V, self.d + 1)

            # Pairwise squared distances (vectorized)
            diff = X[:, None, :] - X[None, :, :]
            dist2 = np.sum(diff**2, axis=2)

            # Gaussian kernel weights and thresholding
            W = np.exp(-dist2 / (2 * sigma**2))
            W[W < cutoff] = 0

            # Remove self-weights (optional, set diag to 0)
            np.fill_diagonal(W, 0.0)

            # Create weighted graph
            G = nx.from_numpy_array(W)

            return G

        self.G = generate_geometric_gaussian_weighted()

        self.L = nx.laplacian_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.random_connection_graph()

class SBMCG(CG):
    """ This class implement a Stochastic Block Model with a consistent connection graph built on top of it

    Parameters
    ----------
    V : int 
        Number of nodes in the network
    d : int 
        Fiber (stalk) dimension
    k : int
        Number of communities
    p_k : np.ndarray
        Discrete distribution over the communities
    p_in : float
        Intra-cluster probability of wiring
    p_out : float
        Inter-cluster probability of wiring
    seed : float
        Seed for randomness
    kron : bool
        Flag for trivial bundle
    consistent : bool
        Flag for consistency
    special : bool
        Flag for special orthogonal group
    """
    def __init__(
        self, 
        V,
        d : int, 
        seed : int,
        k : int,
        p_k : np.ndarray,
        p_in : float,
        p_out : float,
        kron : bool = False,
        consistent : bool = True, 
        special : bool = True
    ) -> None:
        super().__init__(V, d, seed, kron, consistent, special)
        
        # Stochastic block model parameters
        self.k = k                             # Number of communities
        self.p_k = p_k / np.sum(p_k)           # Communities distribution assignment
        self.p_in = p_in                       # Intra cluster wiring probability
        self.p_out = p_out                     # Inter cluster wiring probability

        self.A = np.zeros((V,V))
        
        # Build graph
        self.graph_building()

        # Convert to networkx graph
        self.G = nx.from_numpy_array(self.A)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.random_connection_graph()

    def block_assignment(self):
        """ Assigns node to communities
        """
        self.B = {node: np.random.choice(self.k, p=self.p_k)
                  for node in range(self.V)}

    def block_transition(self):
        """ Defines the transition probability of 
        """
        self.C = np.ones((self.k, self.k)) * self.p_out
        self.C[range(self.k), range(self.k)] = self.p_in

    def edge_generation(self):
        """ Edge sampling routing
        """
        for i in range(self.V):
            for j in range(i + 1, self.V):
                U = self.B[i]
                V = self.B[j]
                prob = self.C[U, V]
                edge = np.random.rand() < prob
                self.A[i, j] = edge
                self.A[j, i] = edge

    def graph_building(self):
        """ Stochastic Block Model setup method calling all routines
        """
        self.block_assignment()
        self.block_transition()
        self.edge_generation()


class GridCG(CG):
    """ This class implements a 2D lattice of a given size with a consistent connection graph built on top of it

    Parameters
    ----------

    side : int
        Length of the size of the lattice: number of nodes would be size**2
    d : int 
        Fiber (stalk) dimension
    w_LB : float
        Lower bound for weights distribution
    w_UB : float
        Upper bound for weights distribution
    kron : bool
        Flag for trivial bundle
    consistent : bool
        Flag for consistency
    special : bool
        Flag for special orthogonal group    
    """

    def __init__(
        self, 
        side : int,
        d : int, 
        w_LB : float = 0.2,
        w_UB : float = 3,
        seed : int = 42,
        kron : bool =False,
        consistent : bool =True, 
        special : bool =True
    ) -> None:
        super().__init__(V = side ** 2, d = d, seed = seed, kron = kron, consitent = consistent, special = special)
        
        self.w_LB = w_LB
        self.w_UB = w_UB

        # Build the 2D grid graph 
        self.G = nx.grid_2d_graph(side, side)

        mapping = {node: idx for idx, node in enumerate(self.G.nodes())}
        self.G = nx.relabel_nodes(self.G, mapping)

        self.A = nx.adjacency_matrix(self.G).toarray()
        self.edges = list(self.G.edges)

        self.W = self.Weights()
        self.L = self.Laplacian()

        self.random_connection_graph()
    
    def weights(
        self
    ) -> np.ndarray:
        """ Sample edge weights from a uniform distribution

        Returns
        -------
        np.ndarray
            Weighted adjacency matrix
        """
        W = np.zeros_like(self.A, dtype=float)
        for edge in self.edges:
            w = np.random.uniform(low=0.2, high=3)
            W[edge[0], edge[1]] = w
            W[edge[1], edge[0]] = w
        return W

    def laplacian(
        self
    ) -> np.ndarray:
        """ Builds the combinatorial laplacian from the adjacency

        Returns
        -------
        np.ndarray
            Combinatorial laplacian matrix
        """
        return np.diag(np.sum(self.W, axis=0)) - self.W