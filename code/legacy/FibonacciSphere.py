
import numpy as np
import random
import networkx as nx
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.linalg import svd
import scipy.sparse as sp

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
    