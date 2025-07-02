import numpy as np
import pandas as pd
import networkx as nx
import electrostatic
from typing import Tuple, List, Optional, Union

class tunnel_class(electrostatic.electrostatic_class):
    """Class for managing single electron tunneling in nanoparticle networks.
    
    This class extends the electrostatic_class to handle single electron tunneling
    events between nanoparticles and electrodes. It manages tunneling junctions,
    calculates tunneling resistances, and handles temperature effects.

    Physical Constants
    ----------------
    ele_charge : float
        Elementary charge [aC]
    kb : float
        Boltzmann constant [eV/K]

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of junctions per nanoparticle
    inv_capacitance_matrix : ndarray
        Inverse of capacitance matrix [1/aF]
    adv_index_rows : ndarray
        Origin nanoparticles (i) in tunneling events i→j
    adv_index_cols : ndarray
        Target nanoparticles (j) in tunneling events i→j
    potential_vector : ndarray
        Potential values for electrodes and nanoparticles [V]
    const_capacitance_values : ndarray
        Capacitance terms for tunneling free energy calculation [aF]

    Methods
    -------
    init_adv_indices()
        Initialize indices for all possible tunneling events
    init_potential_vector(voltage_values)
        Set up potential landscape using electrode voltages
    init_const_capacitance_values()
        Compute capacitance terms for free energy calculations
    return_graph_object()
        Get NetworkX graph representation of the network
    return_potential_vector()
        Get current potential vector
    return_const_capacitance_values()
        Get capacitance terms for free energy calculation
    return_particle_electrode_count()
        Get number of particles and electrodes
    return_advanced_indices()
        Get tunneling event indices
    return_const_temperatures(T)
        Get temperature array for tunneling events
    return_random_resistances(R, Rstd)
        Generate random tunnel resistances
    ensure_undirected_resistances(resistances, average)
        Enforce resistance symmetry R(i,j) = R(j,i)
    update_junction_resistances(resistance_arr, junctions, R)
        Update resistances for specific junctions
    update_nanoparticle_resistances(resistance_arr, nanoparticles, R)
        Update all resistances connected to specific nanoparticles
    """

    def __init__(self, electrode_type: List[str], seed: Optional[int] = None) -> None:
        """Initialize tunneling class.

        Parameters
        ----------
        electrode_type : List[str]
            List specifying electrode types ('constant' or 'floating')
        seed : int, optional
            Random seed for reproducibility, by default None
        """
        super().__init__(electrode_type, seed)
        
        self.ele_charge = 0.160217662  # [aC]
        self.kb = 1.38064852e-5       # [aJ/K]

    def init_adv_indices(self):
        """Initialize indices for all possible tunneling events."""
        # Connection Array to show which nps and electrodes are connected 
        connections = np.zeros((self.N_particles+self.N_electrodes, self.N_junctions+1))
        connections.fill(-100)
        connections[self.N_electrodes:,:] = self.net_topology
        
        # Map electrode connections
        nth_e, nth_np = 1, 0
        while (nth_np < self.N_particles) and (nth_e <= self.N_electrodes):
            if int(self.net_topology[nth_np,0]) == nth_e:
                connections[nth_e-1,1] = nth_np
                nth_e += 1
                nth_np = 0
                continue
            nth_np += 1
        
        connections[:,0] = connections[:,0] - 1
        connections[:,1:] = connections[:,1:] + self.N_electrodes

        # Generate tunneling indices
        adv_index_cols = [list(arr[arr >= 0].astype(int)) for arr in connections]
        adv_index_rows = [len(val)*[i] for i, val in enumerate(adv_index_cols)]
        self.adv_index_cols = np.array([item for sublist in adv_index_cols for item in sublist])
        self.adv_index_rows = np.array([item for sublist in adv_index_rows for item in sublist])

    def init_potential_vector(self, voltage_values: np.ndarray) -> None:
        """Initialize potential landscape using electrode voltages.

        Parameters
        ----------
        voltage_values : np.ndarray
            Electrode voltages [V] as array([V_e1, V_e2, V_e3, ..., V_G])
            where V_G is the gate voltage

        Raises
        ------
        ValueError
            If voltage_values length doesn't match number of electrodes + 1
        """
        if len(voltage_values) != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values")
            
        self.potential_vector = np.zeros(self.N_electrodes + self.N_particles)
        self.potential_vector[0:self.N_electrodes] = voltage_values[:-1]

    def init_const_capacitance_values(self) -> None:
        """Initialize capacitance terms for free energy calculation."""
        row_i = self.adv_index_rows-self.N_electrodes
        col_i = self.adv_index_cols-self.N_electrodes
        row_i2 = (row_i >= 0).astype(int)
        col_i2 = (col_i >= 0).astype(int)
        
        cap_ii = self.inv_capacitance_matrix[row_i, row_i]*row_i2*self.ele_charge*self.ele_charge/2
        cap_jj = self.inv_capacitance_matrix[col_i, col_i]*col_i2*self.ele_charge*self.ele_charge/2
        cap_ij = self.inv_capacitance_matrix[row_i, col_i]*row_i2*col_i2*self.ele_charge*self.ele_charge/2

        self.const_capacitance_values = (cap_ii + cap_jj - 2*cap_ij)

    def return_graph_object(self) -> nx.Graph:
        """
        Returns
        -------
        G : nx.DiGraph
            NetworkX directed graph of the nanoparticle network
        """

        G = nx.DiGraph()
        G.add_nodes_from([i for i in range(self.N_particles+self.N_electrodes)])
        G.add_edges_from([(self.adv_index_rows[i], self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))])

        return G

    def return_potential_vector(self) -> np.ndarray:
        """
        Returns
        -------
        potential_vector : ndarray
            Potential values for electrodes and nanoparticles
        """

        return self.potential_vector

    def return_const_capacitance_values(self) -> np.ndarray:
        """
        Returns
        -------
        const_capacitance_values : ndarray
            Capacitance terms for tunneling free energy calculation
        """
        return self.const_capacitance_values

    def return_particle_electrode_count(self) -> Tuple[int, int]:
        """
        Returns
        -------
        N_particles : int
            Number of nanoparticles
        N_electrodes : int
            Number of electrodes
        """

        return self.N_electrodes, self.N_particles
    
    def return_advanced_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        adv_index_rows : ndarray
            Origin nanoparticles (i) in tunneling events i→j
        adv_index_cols : ndarray
            Target nanoparticles (j) in tunneling events i→j
        """
        return self.adv_index_rows, self.adv_index_cols
    
    def return_const_temperatures(self, T: float = 0.28) -> np.ndarray:
        """
        Parameters
        ----------
        T : float
            Network temperature [K]

        Returns
        -------
        T_arr : ndarray
            Array of temperatures for each tunneling event [eV]
        """
        return np.repeat(T*self.kb, len(self.adv_index_rows))

    def return_random_resistances(self, R: float = 25, Rstd: float = 0) -> np.ndarray:
        """        
        Parameters
        ----------
        R : float
            Average tunnel resistance [MΩ]
        Rstd : float
            Standard deviation of tunnel resistances [MΩ]

        Returns
        -------
        const_R : ndarray
            Variable resistances for each tunneling event i→j sampled from Gaussian
        """

        R_megaO     = R*self.ele_charge*self.ele_charge*1e-12
        R_std_megaO = Rstd*self.ele_charge*self.ele_charge*1e-12
                
        const_R = np.abs(self.rng.normal(R_megaO, R_std_megaO, size=len(self.adv_index_rows)))

        return const_R

    def ensure_undirected_resistances(self, resistances: np.ndarray, average: bool = False) -> np.ndarray:
        """
        Ensures that resistances are undirected by making R(i, j) = R(j, i).
        
        Parameters
        ----------
        resistances : np.ndarray
            1D array of resistance values corresponding to the junction pairs.
        average : bool, optional
            If True, the resistance values for both directions (i, j) and (j, i) are averaged.
            If False, the resistance value of (i, j) overwrites (j, i). Default is False.

        Returns
        -------
        np.ndarray
            Updated resistance values with undirected property enforced.
        """
        # Create a dictionary to map (i, j) to resistance
        pair_to_index = {}
        for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols)):
            pair_to_index[(i, j)] = idx
        
        # Iterate through each resistance and enforce symmetry
        for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols)):
            # Get the reverse pair (j, i)
            reverse_pair = (j, i)
            if reverse_pair in pair_to_index:
                reverse_idx = pair_to_index[reverse_pair]

                # Ensure symmetry
                if average:
                    average_resistance          = (resistances[idx] + resistances[reverse_idx]) / 2.0
                    resistances[idx]            = average_resistance
                    resistances[reverse_idx]    = average_resistance
                else:
                    resistances[reverse_idx]    = resistances[idx]

        return resistances
    
    def update_junction_resistances(self, resistance_arr: np.ndarray, 
                                  junctions: List[Tuple[int, int]], 
                                  R: float = 25) -> np.ndarray:
        """Update tunnel resistances for specific junctions.

        Parameters
        ----------
        resistance_arr : np.ndarray
            Array of current tunnel resistances [MΩ]
        junctions : List[Tuple[int, int]]
            List of (origin, target) junction pairs to update
        R : float, optional
            New resistance value [MΩ], by default 25

        Returns
        -------
        np.ndarray
            Updated resistance array
        """
        R_megaO = R * self.ele_charge * self.ele_charge * 1e-12

        for junc in junctions:
            # Update forward direction
            a = np.where(self.adv_index_rows == junc[0] + self.N_electrodes)[0]
            b = np.where(self.adv_index_cols == junc[1] + self.N_electrodes)[0]
            idx = np.intersect1d(a,b)[0]
            resistance_arr[idx] = R_megaO

            # Update reverse direction
            a = np.where(self.adv_index_cols == junc[0] + self.N_electrodes)[0]
            b = np.where(self.adv_index_rows == junc[1] + self.N_electrodes)[0]
            idx = np.intersect1d(a,b)[0]
            resistance_arr[idx] = R_megaO

        return resistance_arr

    def update_nanoparticle_resistances(self, resistance_arr: np.ndarray, 
                                        nanoparticles: List[int], 
                                        R: float = 25) -> np.ndarray:
        """
        Set all resistances in array of resistances which correspond to jumps TOWARDS a particular nanoparticle 

        Parameters
        ----------
        resistance_arr : np.ndarray
            Array of current tunnel resistances [MΩ]
        nanoparticles : List[int]
            Indices of nanoparticles with new resistance values
        R : float, optional
            New resistance value [MΩ], by default 25

        Returns
        -------
        np.ndarray
            Updated resistance array
        """

        R_megaO = R*self.ele_charge*self.ele_charge*1e-12
        
        for idx in nanoparticles:
            resistance_arr[np.where(self.adv_index_cols == idx + self.N_electrodes)[0]] = R_megaO
            resistance_arr[np.where(self.adv_index_rows == idx + self.N_electrodes)[0]] = R_megaO

        return resistance_arr
    
    def build_conductance_matrix(self, R: np.ndarray):

        src = self.adv_index_rows.copy()
        tgt = self.adv_index_cols.copy()

        np_nodes    = sorted({idx for idx in np.concatenate([src, tgt]) if idx >= 0})
        el_nodes    = sorted({idx for idx in np.concatenate([src, tgt]) if idx <  0})
        N_np        = len(np_nodes)
        N_el        = len(el_nodes)
        total_n     = N_np + N_el

        raw2dense   = {raw: i for i, raw in enumerate(np_nodes + el_nodes)}
        cond_matrix = np.zeros((total_n, total_n))

        for s_raw, t_raw, R_l in zip(src, tgt, R):
            g = 1.0 / R_l
            i = raw2dense[s_raw]
            j = raw2dense[t_raw]

            # symmetric update
            cond_matrix[i, i] += g
            cond_matrix[j, j] += g
            cond_matrix[i, j] -= g
            cond_matrix[j, i] -= g

        return cond_matrix


###########################################################################################################################
###########################################################################################################################

if __name__ == "__main__":

    # Parameter
    N_x, N_y, N_z       = 3,3,1
    electrode_pos       = [[0,0,0],[1,2,0]]
    radius, radius_std  = 10.0, 0.0
    eps_r, eps_s        = 2.6, 3.9
    np_distance         = 1
    voltage_values      = [0.8,0.0,0.0]
    electrode_type      = ['constant','floating']
    high_cap_nps        = [N_x*N_y]
    high_cap            = 1e2

    # Network Initialization
    cubic_system  = tunnel_class(electrode_type)
    cubic_system.cubic_network(N_x, N_y, N_z)
    cubic_system.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    cubic_system.add_np_to_output()
    cubic_system.init_nanoparticle_radius(radius, radius_std)
    cubic_system.update_nanoparticle_radius(high_cap_nps, high_cap)
    cubic_system.calc_capacitance_matrix(eps_r, eps_s, np_distance)
    cubic_system.init_charge_vector(voltage_values)
    cubic_system.init_adv_indices()

    # Return Class Attributes
    topology_arr            = cubic_system.return_net_topology()
    capacitance_matrix      = cubic_system.return_capacitance_matrix()
    inv_capacitance_matrix  = cubic_system.return_inv_capacitance_matrix()
    charge_vector           = cubic_system.return_charge_vector()

    # Print Attributes
    print(cubic_system)
    print("Capacitance Matrix:\n", np.round(capacitance_matrix,2))
    print("Initial Charge Vector:\n", np.round(charge_vector,2))

    # Advanced Indices:
    adv_index_rows, adv_index_cols = cubic_system.return_advanced_indices()

    print("Tunnel Origins:\n", adv_index_rows)
    print("Tunnel Targets:\n", adv_index_cols)