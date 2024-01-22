import numpy as np
import pandas as pd
import networkx as nx
import electrostatic

class tunnel_class(electrostatic.electrostatic_class):
    """
    Class to setup possible charge hopping events
    This class depends on the topology_class and electrostatic_class

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of Junctions per Nanoparticle
    inv_capacitance_matrix : array
        Inverse of capacitance matrix
    tunnel_order : int
        Consider either next neighbor hopping (tunnel_order=1) or also second neighbor hopping (tunnel_order=1)
    adv_index_rows : list
        Nanoparticles i (origin) in tunneling event i to j
    adv_index_cols : list
        Nanoparticles j (target) in tunneling event i to j
    co_adv_index1 : list
        Nanoparticles i (origin) in cotunneling event i to j via k
    co_adv_index2 : list
        Nanoparticles k in cotunneling event i to j via k
    co_adv_index3 : list
        Nanoparticles j (target) in cotunneling event i to j via k
    potential_vector : array
        Potential values for electrodes and nanoparticles
    const_capacitance_values : array
        Sum of capacitance for free energy calculation
    const_capacitance_values_co1 : array
        Sum of capacitance for free energy calculation in cotunneling
    const_capacitance_values_co2 : array
        Sum of capacitance for free energy calculation in cotunneling

    Methods
    -------
    init_adv_indices()
        Initialize indices for numpy broadcasting
    init_potential_vector(voltage_values : np.array)
        Initialize potential landscape via electrode voltages
    init_const_capacitance_values()
        Initialize an array containing C_ii + C_jj + C_ij as parts from inverse capacitance matrix to calculate free energy
    return_potential_vector()
    return_const_capacitance_values()
    return_particle_electrode_count()
    return_advanced_indices()
    return_const_temperatures(T : float)
    return_const_resistances(R)
    return_random_resistances(R=25e-12, Rstd=2e-12)
    -------
    """

    def __init__(self, tunnel_order=1, seed=None)->None:
        """
        Parameters
        ----------
        tunnel_order : int
            Consider either next neighbor hopping (tunnel_order=1) or also second neighbor hopping (tunnel_order=1) 
        seed : int
            Seed for random number generator
        """

        super().__init__(seed)
        
        self.tunnel_order = tunnel_order

        # CONST Parameter
        self.ele_charge = 0.160217662
        self.kb         = 1.38064852e-5

    def init_adv_indices(self):

        if self.tunnel_order >= 1:

            # Connection Array to show which nps and electrodes are connected 
            connections = np.zeros((self.N_particles+self.N_electrodes, self.N_junctions))
            connections.fill(-100)
            connections[self.N_electrodes:,:]   = self.net_topology
            nth_e                               = 1
            nth_np                              = 0
            while ((nth_np < self.N_particles)) and (nth_e <= self.N_electrodes):
                if int(self.net_topology[nth_np,0]) == nth_e:
                    connections[nth_e-1,1] = nth_np
                    nth_e   += 1
                    nth_np  = 0
                    continue
                nth_np += 1
            
            connections[:,0]    = connections[:,0] - 1
            connections[:,1:]   = connections[:,1:] + self.N_electrodes

            # New Indices corresponding to all possible junctions
            adv_index_cols      = [list(arr[arr >= 0].astype(int)) for arr in connections]
            adv_index_rows      = [len(val)*[i] for i, val in enumerate(adv_index_cols)]
            adv_index_cols      = np.array([item for sublist in adv_index_cols for item in sublist])
            adv_index_rows      = np.array([item for sublist in adv_index_rows for item in sublist])
            self.adv_index_rows = adv_index_rows
            self.adv_index_cols = adv_index_cols

            self.co_adv_index1 = []
            self.co_adv_index2 = []
            self.co_adv_index3 = []

        if self.tunnel_order == 2:

            G   = nx.Graph()
            G.add_nodes_from([i for i in range(self.N_particles+self.N_electrodes)])
            G.add_edges_from([(self.adv_index_rows[i],self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))])

            # For each Node
            for node in G.nodes:
                # Get all neighbors
                for neighbor in nx.neighbors(G,node):
                    # Get next neighbors
                    for next_neighbor in nx.neighbors(G,neighbor):
                        event = [node, neighbor, next_neighbor]
                        if not(len(event) > len(set(event))):
                            self.co_adv_index1.append(node)
                            self.co_adv_index2.append(neighbor)
                            self.co_adv_index3.append(next_neighbor)

        self.co_adv_index1  = np.array(self.co_adv_index1, dtype=int)
        self.co_adv_index2  = np.array(self.co_adv_index2, dtype=int)
        self.co_adv_index3  = np.array(self.co_adv_index3, dtype=int)

    def init_potential_vector(self, voltage_values : np.array)->None:
        """
        Initialize potential landscape via electrode voltages

        Parameters
        ----------
        voltage_values : array
           Electrode voltages with [V_e1, V_e2, V_e3, ... V_G]
        """

        self.potential_vector                       = np.zeros(self.N_electrodes+self.N_particles)
        self.potential_vector[0:self.N_electrodes]  = voltage_values[:-1]

    def init_const_capacitance_values(self)->None:
        """
        Initialize an array containing C_ii + C_jj + C_ij as parts from inverse capacitance matrix to calculate free energy
        """

        if self.tunnel_order >= 1:

            row_i   = self.adv_index_rows-self.N_electrodes
            col_i   = self.adv_index_cols-self.N_electrodes
            row_i2  = (row_i >= 0).astype(int)
            col_i2  = (col_i >= 0).astype(int)
            cap_ii  = self.inv_capacitance_matrix[row_i, row_i]*row_i2*self.ele_charge*self.ele_charge/2
            cap_jj  = self.inv_capacitance_matrix[col_i, col_i]*col_i2*self.ele_charge*self.ele_charge/2
            cap_ij  = self.inv_capacitance_matrix[row_i, col_i]*row_i2*col_i2*self.ele_charge*self.ele_charge/2

            self.const_capacitance_values       = (cap_ii + cap_jj - 2*cap_ij)
            self.const_capacitance_values_co1   = np.array([])
            self.const_capacitance_values_co2   = np.array([])
        
        if self.tunnel_order >= 2:

            # First transition
            row_i   = self.co_adv_index1-self.N_electrodes
            col_i   = self.co_adv_index2-self.N_electrodes
            row_i2  = (row_i >= 0).astype(int)
            col_i2  = (col_i >= 0).astype(int)
            cap_ii  = self.inv_capacitance_matrix[row_i, row_i]*row_i2*self.ele_charge*self.ele_charge/2
            cap_jj  = self.inv_capacitance_matrix[col_i, col_i]*col_i2*self.ele_charge*self.ele_charge/2
            cap_ij  = self.inv_capacitance_matrix[row_i, col_i]*row_i2*col_i2*self.ele_charge*self.ele_charge/2

            self.const_capacitance_values_co1 = (cap_ii + cap_jj - 2*cap_ij)

            # Second transition
            row_i   = self.co_adv_index2-self.N_electrodes
            col_i   = self.co_adv_index3-self.N_electrodes
            row_i2  = (row_i >= 0).astype(int)
            col_i2  = (col_i >= 0).astype(int)
            cap_ii  = self.inv_capacitance_matrix[row_i, row_i]*row_i2*self.ele_charge*self.ele_charge/2
            cap_jj  = self.inv_capacitance_matrix[col_i, col_i]*col_i2*self.ele_charge*self.ele_charge/2
            cap_ij  = self.inv_capacitance_matrix[row_i, col_i]*row_i2*col_i2*self.ele_charge*self.ele_charge/2

            self.const_capacitance_values_co2 = (cap_ii + cap_jj - 2*cap_ij)

    def return_potential_vector(self)->np.array:
        """
        Returns
        -------
        potential_vector : array
            Potential values for electrodes and nanoparticles
        """

        return self.potential_vector

    def return_const_capacitance_values(self)->np.array:
        """
        Returns
        -------
        const_capacitance_values : array
            Sum of capacitance for free energy calculation
        const_capacitance_values_co1 : array
            Sum of capacitance for free energy calculation in cotunneling
        const_capacitance_values_co2 : array
            Sum of capacitance for free energy calculation in cotunneling
        """

        return self.const_capacitance_values, self.const_capacitance_values_co1, self.const_capacitance_values_co2
    
    def return_particle_electrode_count(self)->float:
        """
        Returns
        -------
        N_particles : int
            Number of nanoparticles
        N_electrodes : int
            Number of electrodes
        """

        return self.N_electrodes, self.N_particles
    
    def return_advanced_indices(self)->tuple:
        """
        Returns
        -------
        adv_index_rows : list
            Nanoparticles i (origin) in tunneling event i to j
        adv_index_cols : list
            Nanoparticles j (target) in tunneling event i to j
        co_adv_index1 : list
            Nanoparticles i (origin) in cotunneling event i to j via k
        co_adv_index2 : list
            Nanoparticles k in cotunneling event i to j via k
        co_adv_index3 : list
            Nanoparticles j (target) in cotunneling event i to j via k
        """
        
        return self.adv_index_rows, self.adv_index_cols, self.co_adv_index1, self.co_adv_index2, self.co_adv_index3
    
    def return_const_temperatures(self, T=0.28)->np.array:
        """
        Parameter
        ---------
        T : float
            Network temperature

        Returns
        -------
        T_arr : array
            Array of const temperatures for each tunneling event
        """
        
        T_arr = np.repeat(T*self.kb, len(self.adv_index_rows)), np.repeat(T*self.kb, len(self.co_adv_index1))

        return T_arr

    def return_const_resistances(self, R=25e-12)->np.array:
        """        
        Parameter
        ---------
        R : float
            Tunnel resistance

        Returns
        -------
        const_R : array
            Resistances for each tunneling event i to j
        const_R_co1 : array
            Resistances for each cotunneling event i to k
        const_R_co2 : array
            Resistances for each cotunneling event k to j
        """

        if self.tunnel_order >= 1:
            const_R     = np.repeat(R*self.ele_charge*self.ele_charge, len(self.adv_index_rows))
            const_R_co1 = np.array([])
            const_R_co2 = np.array([])

        if self.tunnel_order >= 2:
            const_R_co1 = np.repeat(R*self.ele_charge*self.ele_charge, len(self.co_adv_index2))
            const_R_co2 = np.repeat(R*self.ele_charge*self.ele_charge, len(self.co_adv_index3))

        return const_R, const_R_co1, const_R_co2
    
    def return_random_resistances(self, R=25e-12, Rstd=2e-12):
        """        
        Parameter
        ---------
        R : float
            Average Tunnel resistance
        R_std : float
            Standard deviation of tunnel resistances

        Returns
        -------
        const_R : array
            Variable Resistances for each tunneling event i to j sample from Gaus
        const_R_co1 : array
            Empty Array
        const_R_co2 : array
            Empty Array
        """
                
        if self.tunnel_order >= 1:
            const_R     = np.repeat(np.abs(np.random.normal(R*self.ele_charge*self.ele_charge, Rstd)), len(self.adv_index_rows))
            const_R_co1 = np.array([])
            const_R_co2 = np.array([])

        if self.tunnel_order >= 2:
            "Random resistance not supported in tunnel_order >= 2"

        return const_R, const_R_co1, const_R_co2
    
    def set_np_resistance(self, np_index, np_R, resistance_arr):
        """
        Set all resistances in array of resistances which correspond to jumps TOWARDS a particular nanoparticle 

        Parameter
        ---------
        np_index : int
            Index of nanoparticle with new resistance value
        np_R : float
            new resistance value
        resistance_arr : array
            Resistances for each tunneling event

        Returns
        -------
        resistance_arr : array
            Resistances for each tunneling event
        """
        
        resistance_arr[np.where(self.adv_index_cols==np_index)[0]] = np_R

        return resistance_arr

###########################################################################################################################
###########################################################################################################################

if __name__ == "__main__":

    # Parameter
    N_x, N_y, N_z       = 3,3,1
    electrode_pos       = [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]
    radius, radius_std  = 10.0, 0.0
    eps_r, eps_s        = 2.6, 3.9
    np_distance         = 1
    voltage_values      = [0.1,0.2,-0.1,0.3,-0.8]
    tunnel_order        = 1

    # Cubic Network Initialization
    cubic_system  = tunnel_class(tunnel_order)
    cubic_system.cubic_network(N_x, N_y, N_z)
    cubic_system.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    cubic_system.attach_np_to_gate()
    cubic_system.init_nanoparticle_radius(radius, radius_std)
    cubic_system.calc_capacitance_matrix(eps_r, eps_s, np_distance)
    cubic_system.init_charge_vector(voltage_values)
    cubic_system.init_adv_indices()

    # Return Class Attributes
    topology_arr            = cubic_system.return_net_topology()
    capacitance_matrix      = cubic_system.return_capacitance_matrix()
    inv_capacitance_matrix  = cubic_system.return_inv_capacitance_matrix()
    charge_vector           = cubic_system.return_charge_vector()

    # Print Attributes
    print("Cubic Network Topology:\n",      topology_arr)
    print("Capacitance Matrix:\n",          capacitance_matrix)
    print("Inverse Capacitance Matrix:\n",  inv_capacitance_matrix)
    print("Initial Charge Vector:\n",       charge_vector)

    # Advanced Indices:
    adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2, co_adv_index3 = cubic_system.return_advanced_indices()

    print("Tunnel Origins:\n", adv_index_rows)
    print("Tunnel Targets:\n", adv_index_cols)