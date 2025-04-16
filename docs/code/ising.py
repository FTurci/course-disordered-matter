import numpy as np
import tqdm
from numpy.random import rand
import matplotlib.pyplot as plt

class IsingModel:
    def __init__(self, N, equilibration=1024, production=1024):
        """
        Initialize the Ising model simulation.
        
        Parameters:
        -----------
        N : int
            Size of the lattice (N x N)
        equilibration : int
            Number of Monte Carlo sweeps for equilibration
        production : int
            Number of Monte Carlo sweeps for calculation


        Example usage:

        # Initialize model
        model = IsingModel(N=16, nt=88, equilibration=1024, production=1024)
        
        # Run the simulation
        model.simulate()
        
        # Plot the results
        model.plot_results()    
        """
        self.N = N
        self.equilibration = equilibration
        self.production = production
        
        
        # Normalization factors
        self.n1 = 1.0/(production*N*N)
        self.n2 = 1.0/(production*production*N*N)
    
    def initialstate(self):
        """ Generate a random spin configuration for initial condition """
        np.random.seed(1234)  # For reproducibility
        state = 2*np.random.randint(2, size=(self.N, self.N),dtype=np.int32)-1
        return state
    
    def vanilla_mcmove(self, config, beta):
        """Optimized Metropolis update (minimal numpy overhead)"""
        N = self.N
        for _ in range(N * N):
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = config[a, b]
            nb = (
                config[(a + 1) % N, b]
                + config[a, (b + 1) % N]
                + config[(a - 1) % N, b]
                + config[a, (b - 1) % N]
            )
            cost = 2 * s * nb
            if cost <= 0 or rand() < self.exp_cache.get(cost, np.exp(-cost * beta)):
                config[a, b] = -s
        return config
    import numpy as np

    def numpy_mcmove(self, config, beta):
        """Optimized Metropolis update using numpy without the cache."""
        N = self.N
        # Generate random indices for updating sites
        indices = np.random.randint(0, N, size=(2, N * N))
        a, b = indices[0], indices[1]
        
        # Extract the current spin configuration at those indices
        s = config[a, b]

        # Get the neighbors efficiently with numpy slicing and periodic boundary conditions
        nb = (
            config[(a + 1) % N, b] +   # right
            config[a, (b + 1) % N] +   # down
            config[(a - 1) % N, b] +   # left
            config[a, (b - 1) % N]     # up
        )

        # Calculate the cost (2 * s * nb) for all selected indices at once
        cost = 2 * s * nb
        
        # Compute the exponential for the Metropolis criterion
        exp_vals = np.exp(-cost * beta)

        # Generate random values for the Metropolis acceptance criterion
        rand_vals = np.random.rand(N * N)

        # Accept flips based on the Metropolis criterion
        accept_flip = (cost <= 0) | (rand_vals < exp_vals)

        # Apply the flips to the configuration
        config[a[accept_flip], b[accept_flip]] = -s[accept_flip]
        
        return config


    def calcEnergy(self, config):
        """ Energy of a given configuration """
        energy = 0
        N = self.N
        for i in range(N):
            for j in range(N):
                S = config[i,j]
                nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
                energy += -nb*S
        return energy/4.
    
    def calcMag(self, config):
        """ Magnetization of a given configuration """
        return np.sum(config)
    
    def simulate(self, temperature=1.0):
        """ Run the simulation for all temperature points """
        self.exp_cache = {2*d: np.exp(-2*d/temperature) for d in range(5)}
        T = temperature
        E1 = M1 = E2 = M2 = 0
        config = self.initialstate()
        iT = 1.0/T
        iT2 = iT*iT
        
        beginning = config.copy()
        # Equilibration phase
        print("equilibration")
        for i in tqdm.tqdm(range(self.equilibration)):
            self.numpy_mcmove( config, iT)

        
        print("production")
        # Measurement phase
        for i in range(self.production):
            self.numpy_mcmove( config, iT)
            self.config = config
        
    
    def plot_config(self):
        """ Plot the results of the simulation """
        fig,ax = plt.subplots(figsize=(5, 5))
        ax.matshow(self.config, cmap='copper') 
        ax.axis('off')   
        plt.tight_layout()
        plt.show()