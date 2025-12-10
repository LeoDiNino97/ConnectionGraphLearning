import numpy as np 
import wandb 
from tqdm import tqdm

from utils.solver_utils import * 

class SCGL:
    """ Implementation of the proposed Structured Connection Graph Learning method

    Parameters
    ----------

    V : int 
        Number of nodes
    d : int
        Stalk dimension
    k : int 
        Number of connected components (spectral prior)
    alpha : float
        Hyperparameter regulating sparsity
    beta : float 
        Hyperparameter regulating consistency
    gamma : float
        Hyperparameter regulating reconstruction error
    eps : float
        Numerical stability control
    initialization_mode : str
        Mode of initialization 
    initialization_seed : int
        Seed for initialization
    max_init_its : int
        Maximum number of iterations for the initialization
    w_inits : np.ndarray
        Edge weights initialization if given
    O_inits : np.ndarray
        Node basis initialization if given
    max_w_its : int
        Maximumm number of descent step at each episode for w
    proximal_mode : str
        Sparsity inducing penalty mode
    exact_linesearch : bool
        Whether to perform Armijo update in w descend
    update_frames : bool
        Flag for learning node bases or not
    max_O_its : int
        Maximumm number of descent step at each episode for O
    SOC : bool
        Whether to run in SOC mode or RGD mode for update in O
    rho : float
        Hyperparameter regulating strength of convexity in SOC routine
    R_solver : str
        Riemannian solver
    bases : dict
        Input bases if given
    noisy : bool
        Flag for noise in learning
    c1 : float
        Lower bound on eigenvalues
    c2 : float
        Upper bound on eigenvalues
    beta_factor : float
        Beta scheduling multiplier
    fix_beta : bool
        Flag for beta scheduling
    beta_min : float
        Lower bound for beta scheduling
    beta_max : float
        Upper bound for beta scheduling
    rel_tol : float
        Relative tolerance to declare convergence
    abs_tol : float
        Absolute tolerance to declare convergence
    loss_tol : float
        Tolerance over loss decreasing to restart patience mechanism
    patience : int
        Number of iterations before convergence is stated if loss does not decrease enough
    MAX_ITER : int
        Maximum number of iterations
    verbose : bool
        Flag for verbosity
    """
    def __init__(
        self,
        V : int,
        d : int,
        k : int, 
        alpha : float,
        beta : float, 
        gamma : float,
        eps : float,
        initialization_mode : str,
        initialization_seed : int,
        max_init_its : int,
        w_inits : np.ndarray,
        O_inits : np.ndarray,
        max_w_its : int,
        proximal_mode : str,
        exact_linesearch : bool,
        update_frames : bool,
        max_O_its : int,
        SOC : bool,
        rho : float,
        R_solver : str,
        bases : dict,
        noisy : bool,
        c1 : float,
        c2 : float,
        beta_factor : float,
        fix_beta : bool,
        beta_min : float,
        beta_max : float,
        rel_tol : float,
        abs_tol : float,
        loss_tol : float,
        patience : int,
        MAX_ITER : int,
        verbose : bool
    ): 
        # System dimension
        self.V = V
        self.d = d
        self.k = k

        # Hyperparameter
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

        # Initialization setup
        self.initialization_mode = initialization_mode
        self.initialization_seed = initialization_seed
        self.max_init_its = max_init_its
        self.w_inits = w_inits
        self.O_inits = O_inits

        # w update setup
        self.max_w_its = max_w_its
        self.proximal_mode = proximal_mode
        self.exact_linesearch = exact_linesearch

        # O update setup
        self.update_frames = update_frames
        self.max_O_its = max_O_its
        self.SOC = SOC
        self.rho = rho 
        self.R_solver = R_solver
        self.bases = bases

        # Noisy or not
        self.noisy = noisy

        # Bounds on spectrum
        self.c1 = c1
        self.c2 = c2

        # Beta schedule
        self.beta_factor = beta_factor
        self.fix_beta = fix_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        
        # Convergence parameters
        self.rel_tol = rel_tol
        self.abs_tol = abs_tol
        self.loss_tol = loss_tol
        self.patience = patience 
        self.MAX_ITER = MAX_ITER

        # Verbosity
        self.verbose = verbose

    
    def SCGL_initialization(
        self,
        X : np.ndarray,
    ):
        """ Initialization routine

        Parameters
        ----------

        X : np.ndarray
            Observed signals
        """

        S = X @ X.T / X.shape[1]
        Z = np.copy(X)

        # Initialization of the block w
        if self.verbose:
            print('Initializing w and O...')

        if self.w_inits is None or self.O_inits is None:
            init_args = Initialization(
                M = X.shape[1],
                S = S, 
                d = self.d, 
                V = self.V,
                noisy = self.noisy,
                beta_0 = self.k, 
                mode = self.initialization_mode,
                MAX_ITER = self.max_init_its, 
                verbosity = self.verbose,
                bases = self.bases,
                seed = self.initialization_seed
            )
            
            w = init_args[0]
            O = init_args[1]

            if self.noisy:
                sigma_2_hat = init_args[2]
                self.gamma = 1 / (2 * sigma_2_hat)
            else:
                self.gamma = 1

        else:
            w = self.w_inits
            O = self.O_inits
            if self.noisy:
                sigma_2_hat = np.mean(np.linalg.eigvalsh(X @ X.T / X.shape[1])[0 : self.d * self.k])
                self.gamma = 1 / (2 * sigma_2_hat)
            else:
                self.gamma = 1

        if self.noisy:
            Z = Update_Z(w, O, self.gamma, X, self.V, self.d)
        
        # First call of each update method is to initialize the other blocks of variables
        if self.verbose:
            print('Initializing U...')

        U = Update_U(
            w = w,
            k = self.k,
            V = self.V,
            d = self.d        
        )

        if self.verbose:
            print('Initializing lambda...')

        lambda_ = Update_Lambda(
            U = U, 
            w = w, 
            beta = self.beta, 
            gamma = self.gamma,
            c1 = self.c1, 
            c2 = self.c2, 
            V = self.V, 
            k = self.k, 
            d = self.d
            )
        
        return w, U, S, X, Z, O, lambda_
    
    def SCGL_main_loop(
        self,
        w : np.ndarray,
        U : np.ndarray,
        S : np.ndarray,
        X : np.ndarray,
        Z : np.ndarray,
        O : np.ndarray,
        lambda_ : np.ndarray,
    ) -> None:
        """ Main optimization loop
        """
        # Preallocating loss
        loss = np.zeros(self.MAX_ITER)
        S = Z @ Z.T / Z.shape[1]
        
        if self.proximal_mode != 'ReweightedL1':
            K = np.copy(S)
        else:
            H = self.alpha * (np.eye(self.d * self.V) - np.ones((self.d * self.V, self.d * self.V)))
            K = S + H

        # Initializing patience counter
        plateau_counter = 0

        # Handle initialization only baselines:
        if self.MAX_ITER == 0: 
            return O, w, None,
        else:
            for t in tqdm(range(self.MAX_ITER)):
                # Update Z
                if self.noisy:
                    Z_hat = Update_Z(
                        w = w,
                        O = O,
                        gamma = self.gamma, 
                        X = X,
                        V = self.V,
                        d = self.d
                    )

                else:
                    Z_hat = X
        
                S = Z_hat @ Z_hat.T / Z_hat.shape[1]

                # Update w
                w_hat = Update_w(
                    w = w, 
                    U = U, 
                    S = S, 
                    O = O, 
                    lambda_ = lambda_, 
                    alpha = self.alpha, 
                    beta = self.beta, 
                    gamma = self.gamma,
                    V = self.V, 
                    d = self.d, 
                    its = self.max_w_its, 
                    proximal_mode = self.proximal_mode, 
                    exact_linesearch = self.exact_linesearch
                )
                
                # Update O
                if self.update_frames == True:
                    if not self.SOC:
                        O_hat = Update_O_RG(
                            O = O, 
                            S = S,
                            w = w_hat, 
                            V = self.V, 
                            d = self.d, 
                            lambda_ = lambda_, 
                            beta = self.beta, 
                            O_init = True,
                            max_its = self.max_O_its,
                            bases = self.bases,
                            solver = self.R_solver
                        )
                    else:
                        O_hat = Update_O_SOC(
                            O = O,
                            Z = Z_hat,
                            w = w_hat,
                            V = self.V,
                            d = self.d,
                            rho = self.rho,
                            MAX_ITER = self.max_O_its,
                            abs_tol = self.abs_tol,
                            rel_tol = self.rel_tol
                        )

                else:
                    O_hat = O

                # Update U
                U_hat = Update_U(
                    w = w_hat, 
                    k = self.k, 
                    V = self.V, 
                    d = self.d,
                )

                # Update lambda
                lambda_hat = Update_Lambda(
                    U = U_hat, 
                    w = w_hat, 
                    beta = self.beta, 
                    gamma = self.gamma,
                    c1 = self.c1, 
                    c2 = self.c2, 
                    V = self.V, 
                    k = self.k, 
                    d = self.d
                )

                if self.proximal_mode == 'ReweightedL1':
                    K = S + H / (- O_hat.T @ LKron(w, self.V, self.d) @ O_hat + self.eps)
                else:
                    K = S

                # Routine of scheduling for beta
                if not self.fix_beta:

                    n_zero_eigenvalues = np.sum(np.isclose(np.abs(np.linalg.eigvalsh(LKron(w_hat, self.V, self.d))), 0, atol = 1e-9))
                    if self.k * self.d < n_zero_eigenvalues:
                        if self.verbose:
                            print('Increasing beta...', self.beta)
                        self.beta *= (1 + self.beta_factor)

                    elif self.k * self.d > n_zero_eigenvalues:
                        if self.verbose:
                            print('Decreasing beta...')
                        self.beta /= (1 + self.beta_factor)

                    if self.beta < self.beta_min:
                        self.beta = self.beta_min

                    if self.beta > self.beta_max:
                        self.beta = self.beta_max

                # Convergence check
                converged = True

                # Primal residuals on w
                w_err = np.abs(w - w_hat)
                converged_w = np.all(w_err <= 0.5 * self.rel_tol * (w + w_hat)) or np.all(w_err <= self.abs_tol)
                converged *= converged_w

                # Variables update
                w = w_hat
                Z = Z_hat

                if self.update_frames == True:
                    O = O_hat

                U = U_hat
                lambda_ = lambda_hat

                if converged:

                    print(f'Convergence reached in {t} iterations on the residuals')
                    break

                # Loss storing 
                loss[t] = loss_(
                    V = self.V,
                    d = self.d, 
                    X = X,
                    Z = Z, 
                    U = U,
                    O = O,
                    w = w, 
                    S = S,
                    lambda_ = lambda_,
                    gamma = self.gamma,
                    beta = self.beta,
                    alpha = self.alpha, 
                    noisy = self.noisy
                )

                # ---- W&B logging block ----
                wandb.log({
                    "loss": loss[t],
                    "iteration": t,
                    "beta": self.beta,
                    "max_w_update": float(np.max(np.abs(w - w_hat))),
                }, step=t
                )
                # ---------------------------

                if t > 0 :

                    # Patience mechanism on loss plateaus
                    relative_loss_change = np.abs(loss[t] - loss[t - 1]) / (np.abs(loss[t - 1]) + 1e-8)
                    if relative_loss_change < self.loss_tol:
                        plateau_counter += 1
                        if plateau_counter >= self.patience:
        
                            print(f'Convergence assumed on the loss plateau at iteration {t}')
                            break
                    else:
                        plateau_counter = 0

        return O, w, Z, U, lambda_, loss[:t+1]
    
    def fit(
        self,
        X : np.ndarray
    ): 
        """ Single learning call
        """

        # Start wandb run
        wandb.init(
            project="SCGL",
            name=f"SCGL_run_V{self.V}_d{self.d}",
            config={
                "V": self.V,
                "d": self.d,
                "k": self.k,
                "alpha": self.alpha,
                "beta_initial": self.beta,
                "gamma": self.gamma,
                "proximal_mode": self.proximal_mode,
                "update_frames": self.update_frames,
                "SOC": self.SOC,
                "rel_tol": self.rel_tol,
                "abs_tol": self.abs_tol,
                "loss_tol": self.loss_tol,
                "patience": self.patience,
                "MAX_ITER": self.MAX_ITER,
            }
        )

        init_args = self.SCGL_initialization(X)
        O, w, Z, _, _, loss_log = self.SCGL_main_loop(
            init_args[0], 
            init_args[1], 
            init_args[2], 
            init_args[3], 
            init_args[4], 
            init_args[5], 
            init_args[6]
        )
        
        wandb.finish()

        return O, w, Z, loss_log
