import networkx as nx
import numpy as np
from sklearn.datasets import make_spd_matrix as make_spd_matrix
import autograd.numpy as anp
import pandas as pd
import pickle

interim_data_dir="../../data/interim/"
important_data_dir="../../data/important/"
raw_data_dir="../../data/raw/"
figs_data_dir="../../figs/"

def save_obj(obj, name, data_dir):
    with open(data_dir+name+'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name, data_dir):
    with open(data_dir+name+'.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_obj_parquet(obj, name, data_dir):
    """
    This function saves an object using parquet. (This is not sensitive to versioning.)

    INPUT
    =====
    obj: list/pd.dataframe/np.array. Object to be saved.
    name: str. Name for the file (without the extension).
    data_dir: str. Path where to place the file.
    """
    
    if isinstance(obj, list):
        df = pd.DataFrame(obj)
        df.to_parquet(data_dir + name + '.parquet')
    elif isinstance(obj, pd.DataFrame):
        obj.to_parquet(data_dir + name + '.parquet')
    elif isinstance(obj, np.ndarray):
        df = pd.DataFrame(obj)
        df.to_parquet(data_dir + name + '.parquet')
    else:
        raise ValueError("Unsupported data type")
    
def load_obj_parquet(name, data_dir):
    """
    This function loads an object using parquet.
    
    INPUT
    =====
    name: str. Name for the file (without the extension).
    data_dir: str. Path where to find the file.
    """
    try:
        df = pd.read_parquet(data_dir + name + '.parquet')
        if df.index.name is not None:
            df.reset_index(inplace=True)
        return df
    except FileNotFoundError:
        raise FileNotFoundError("Parquet file not found")

def stiefel_matrix(n_rho, n_tau, lb=-1., ub=1., SO=False, seed=42):
    """
    This function samples an extension map from the Stiefel manifold.

    INPUT
    =====
    - n_rho: int. Dimensionality of the stalk on the node, strictly positive.
    - n_tau: int. Dimensionality of the costalk on the edge, strictly positive.
    - seed: int. anp seed for the random module, strictly positive.
    - lb: int. Lower bound for the restriction map coefficients
    - ub: int. Upper bound for the restriction map coefficients
   
    OUTPUT
    ======
    - F_rhotau_t: anp.array, shape (n_rho,n_tau). Transpose of the restriction map. 
    """
    
    assert n_rho>=n_tau, "The dimensionality of stalks on the nodes (n_rho) must be >= than that on the edges (n_tau)."

    anp.random.seed(seed)
    V = anp.random.uniform(lb,ub,size=(n_rho, n_tau))
    F_rhotau_t, _ = anp.linalg.qr(V)

    if SO and anp.linalg.det(F_rhotau_t)<0:
        F_rhotau_t[:,0]*=-1.

    return F_rhotau_t

def stiefel_arc_length(A,B):
    """According to pg. 30 in 
    Edelman, Alan, Tomás A. Arias, and Steven T. Smith. "The geometry of algorithms with orthogonality constraints." SIAM journal on Matrix Analysis and Applications 20.2 (1998): 303-353.
    """
    _, S, _ = anp.linalg.svd(A.T@B)
    S = np.round(S, 3)
    theta = anp.nan_to_num(anp.arccos(S))
    return anp.linalg.norm(theta, ord=2)

def frobenious_abs_distance(V_pred, V_true, norm=True):
    if norm:
        return anp.linalg.norm(abs(V_true)- abs(V_pred))/anp.linalg.norm(abs(V_true))
    return anp.linalg.norm(abs(V_true)- abs(V_pred))


def generate_connection_graph(G, n, consistency=True):
    """
    Generate a connection graph \mathbb{G} from an input undirected graph G. THe dimension of the fibers is n.
    
    INPUT
    =====
    - G: networkx.Graph, the input undirected graph.
    - n: int, the dimension of the fibers.
    - consistency: bool, whether to generate a consistent connection graph.
    
    OUTPUT
    ======
    - dict, with adjacency, degree, incidence, and Laplacian matrices.
    """
    
    if not nx.is_connected(G):
        num_components = nx.number_connected_components(G)
        print(f"Warning: The input graph is not connected. It has {num_components} connected components.")
    
    num_nodes = G.number_of_nodes()

    weights = {}
    O_matrices = {}
    node_frames = {}
    
    #Assign weights from graph (default to 1 if not provided)
    for u, v, data in G.edges(data=True):
        weights[(u, v)] = weights[(v, u)] = data.get("weight", 1.0)
        
    #Generate orthogonal matrices
    if consistency:
        #Generate random SO(n) matrices for each node
        for node in G.nodes():
            node_frames[node] = stiefel_matrix(n, n, SO=True, seed=anp.random.randint(10000))
        
        #Define edge matrices based on node frames
        for u, v in G.edges():
            O_matrices[(u, v)] = node_frames[u].T @ node_frames[v]
            O_matrices[(v, u)] = O_matrices[(u, v)].T  
    else:
        #Assign random matrices in SO(n) independently for each edge
        for u, v in G.edges():
            Q = stiefel_matrix(n, n, SO=True, seed=anp.random.randint(10000))
            O_matrices[(u, v)] = Q
            O_matrices[(v, u)] = Q.T 
    
    #Connection adjacency matrix (block matrix)
    A_conn = anp.zeros((num_nodes * n, num_nodes * n))
    for u, v in G.edges():
        i, j = u * n, v * n
        A_conn[i:i+n, j:j+n] = weights[(u, v)] * O_matrices[(u, v)]
        A_conn[j:j+n, i:i+n] = weights[(v, u)] * O_matrices[(v, u)]
    
    # Connection degree matrix (block diagonal)
    D_conn = anp.zeros_like(A_conn)
    for u in G.nodes():
        i = u * n
        d_u = sum(weights[(u, v)] for v in G.neighbors(u))
        D_conn[i:i+n, i:i+n] = d_u * anp.eye(n)
    
    # Connection Laplacian
    L_conn = D_conn - A_conn
    
    return {
        'w_uv': weights,
        'O_uv': O_matrices,
        'O_u': node_frames,
        'A': A_conn,
        'D': D_conn,
        'L': L_conn
    }

def add_weights_to_graph(G, weight_type="constant", weight_value=1.0):
    """
    Generate a weighted bipartite graph with two sets of nodes.
    
    INPUT
    =====
    - G: networkx.Graph, the input undirected graph.
    - weight_type: str, 'constant' or 'random' to set edge weights.
    - weight_value: float, the constant weight value (default: 1.0).
    
    OUTPUT
    ======
    - G: networkx.Graph, the weighted undirected graph.
    """
    assert weight_type in ["constant", "random"], "Unknown value for 'weight_type'"

    for u, v in G.edges():
        if weight_type == "random":
            G[u][v]['weight'] = np.random.uniform(0, 1)  
        else:
            G[u][v]['weight'] = weight_value  
    
    return G