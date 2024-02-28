# Imports
import os
import numpy as np
import fenics as fn
import logging
from numba.experimental import jitclass
from numba import int64, float64, boolean, types

class DopantNet():

    def __init__(self, N_a, N_d, electrodes, static_electrodes, mu = 0, I_0=100, a=0.25) -> None:
        
        # Base Parameter
        self.nu     = 1                 # Hop attempt frequency [1/s]
        self.kT     = 1                 # Temperature energy
        self.I_0    = I_0*self.kT       # Interaction energy
        self.time   = 0                 # Time [s]
        self.mu     = mu                # Equilibrium chemical potential
        self.N_a    = N_a               # Number of acceptors
        self.N_d    = N_d               # Number of donors
        self.R      = self.N_a**(-1/2)  # Average distance between acceptors
        self.ab     = a*self.R          # Bohr radius/localization radius
        self.res    = 1/100             # Resolution used for solving the chemical potential profile

        # Electrodes
        self.electrodes         = electrodes
        self.P                  = self.electrodes.shape[0]
        self.static_electrodes  = static_electrodes


        # Attributes
        self.transitions            = np.zeros((self.N_a + self.P,
                                                self.N_a + self.P))
        self.transitions_constant   = np.zeros((self.N_a + self.P,
                                                self.N_a + self.P))
        self.distances              = np.zeros((self.N_a + self.P,
                                                self.N_a + self.P))
        self.vectors                = np.zeros((self.N_a + self.P,
                                                self.N_a + self.P, 3))
        self.site_energies          = np.zeros((self.N_a + self.P,))
        self.problist               = np.zeros((self.N_a+self.P)**2)
        self.occupation             = np.zeros(self.N_a, dtype=bool)
        self.electrode_occupation   = np.zeros(self.P, dtype=int)

        self.place_dopants_random()
        self.place_charges_random()
        self.calc_distances()
        self.calc_transitions_constant()
        self.init_V()
        self.calc_E_constant_V_comp()

    def place_dopants_random(self, seed_a=None, seed_d=None):

        if ((seed_a == None) or (seed_d == None)):

            self.acceptors  = np.random.rand(self.N_a, 3)
            self.donors     = np.random.rand(self.N_d, 3)
        
        else:

            rs0             = np.random.RandomState(seed_a)
            rs1             = np.random.RandomState(seed_d)
            self.acceptors  = rs0.rand(self.N_a, 3)
            self.donors     = rs1.rand(self.N_d, 3)
        
    def place_charges_random(self):

        self.occupation = np.zeros(self.N_a, dtype=bool)
        charges_placed  = 0

        while(charges_placed < self.N_a-self.N_d):

            trial = np.random.randint(self.N_a)

            if(self.occupation[trial] == False):

                self.occupation[trial]  = True
                charges_placed          += 1
    
    def calc_distances(self):
        
        for i in range(self.N_a+self.P):

            for j in range(self.N_a+self.P):

                if(i is not j):

                    # Distance electrode -> electrode
                    if(i >= self.N_a and j >= self.N_a):
                        self.distances[i, j]    = self.dist(self.electrodes[i - self.N_a, :3],
                                                            self.electrodes[j - self.N_a, :3])
                        self.vectors[i, j]      = ((self.electrodes[j - self.N_a, :3]
                                                    - self.electrodes[i - self.N_a, :3])
                                                    /self.distances[i, j])

                    # Distance electrodes -> acceptor
                    elif(i >= self.N_a and j < self.N_a):
                        self.distances[i, j]    = self.dist(self.electrodes[i - self.N_a, :3],
                                                            self.acceptors[j])
                        self.vectors[i, j]      = ((self.acceptors[j]
                                                - self.electrodes[i - self.N_a, :3])
                                                /self.distances[i, j])
                        
                    # Distance acceptor -> electrode
                    elif(i < self.N_a and j >= self.N_a):
                        self.distances[i, j]    = self.dist(self.acceptors[i],
                                                          self.electrodes[j - self.N_a, :3])
                        self.vectors[i, j]      = ((self.electrodes[j - self.N_a, :3]
                                                - self.acceptors[i])
                                                /self.distances[i, j])
                        
                    # Distance acceptor -> acceptor
                    elif(i < self.N_a and j < self.N_a):
                        self.distances[i, j]    = self.dist(self.acceptors[i],
                                                            self.acceptors[j])
                        self.vectors[i, j]      = ((self.acceptors[j]
                                                    - self.acceptors[i])
                                                    /self.distances[i, j])
                    
    def calc_transitions_constant(self):
        
        self.transitions_constant   = self.nu*np.exp(-2 * self.distances/self.ab)
        self.transitions_constant   -= np.eye(self.transitions.shape[0])
    
    def init_V(self):
        '''
        This function sets up various parameters for the calculation of
        the chemical potential profile using fenics.
        It is generally assumed that during the simulation of a 'sample'
        the following are unchanged:
        - dopant positions
        - electrode positions/number
        Note: only 2D support for now
        '''
        # Turn off log messages
        fn.set_log_level(logging.WARNING)

        # Put electrode positions and values in a dict
        self.fn_electrodes = {}

        for i in range(self.P):
            self.fn_electrodes[f'e{i}_x']   = self.electrodes[i, 0]
            self.fn_electrodes[f'e{i}_y']   = self.electrodes[i, 1]
            self.fn_electrodes[f'e{i}']     = self.electrodes[i, 3]

        for i in range(self.static_electrodes.shape[0]):
            
            self.fn_electrodes[f'es{i}_x']  = self.static_electrodes[i, 0]
            self.fn_electrodes[f'es{i}_y']  = self.static_electrodes[i, 1]
            self.fn_electrodes[f'es{i}']    = self.static_electrodes[i, 3]

        self.fn_expression  = ''
        surplus              = 1/10  # Electrode modelled as point +/- surplus

        for i in range(self.P):
            
            if(self.electrodes[i, 0] == 0 or self.electrodes[i, 0] == 1):
                self.fn_expression += (f'x[0] == e{i}_x && '
                                        f'x[1] >= e{i}_y - {surplus} && '
                                        f'x[1] <= e{i}_y + {surplus} ? e{i} : ')
            else:
                self.fn_expression += (f'x[0] >= e{i}_x - {surplus} && '
                                        f'x[0] <= e{i}_x + {surplus} && '
                                        f'x[1] == e{i}_y ? e{i} : ')
        for i in range(self.static_electrodes.shape[0]):
            
            if(self.static_electrodes[i, 0] == 0 or self.static_electrodes[i, 0] == 1):
                self.fn_expression += (f'x[0] == es{i}_x && '
                                        f'x[1] >= es{i}_y - {surplus} && '
                                        f'x[1] <= es{i}_y + {surplus} ? es{i} : ')
            else:
                self.fn_expression += (f'x[0] >= es{i}_x - {surplus} && '
                                        f'x[0] <= es{i}_x + {surplus} && '
                                        f'x[1] == es{i}_y ? es{i} : ')

        self.fn_expression += f'{self.mu}'  # Add constant chemical potential

        # Define boundary expression
        self.fn_boundary = fn.Expression(self.fn_expression,
                                         degree = 1,
                                         **self.fn_electrodes)

        self.fn_mesh = fn.RectangleMesh(fn.Point(0, 0),
                                        fn.Point(1, 1),
                                        int(1//self.res),
                                        int(1//self.res))

        # Define function space
        self.fn_functionspace = fn.FunctionSpace(self.fn_mesh, 'P', 1)

        # Define fenics boundary condition
        self.fn_bc = fn.DirichletBC(self.fn_functionspace,
                                    self.fn_boundary,
                                    self.fn_onboundary)

        # Write problem as fn_a == fn_L
        self.V      = fn.TrialFunction(self.fn_functionspace)
        self.fn_v   = fn.TestFunction(self.fn_functionspace)
        self.fn_a   = fn.dot(fn.grad(self.V), fn.grad(self.fn_v)) * fn.dx
        self.fn_f   = fn.Constant(0)
        self.fn_L   = self.fn_f*self.fn_v*fn.dx

        # Solve V
        self.V              = fn.Function(self.fn_functionspace)
        fn.solve(self.fn_a  == self.fn_L, self.V, self.fn_bc)  
    
    def calc_E_constant_V_comp(self):
        
        # Initialization
        self.eV_constant    = np.zeros((self.N_a,))
        self.comp_constant  = np.zeros((self.N_a,))

        for i in range(self.N_a):
            
            self.eV_constant[i]     += self.V(self.acceptors[i, 0], self.acceptors[i, 1])

            # Add compensation
            self.comp_constant[i]   += self.I_0*self.R* sum(
                                        1/self.dist(self.acceptors[i], self.donors[k]) for k in range(self.N_d))
            
        self.E_constant = self.eV_constant + self.comp_constant

        # Calculate electrode energies
        self.site_energies[self.N_a:] = self.electrodes[:, 3]