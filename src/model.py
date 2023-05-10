import numpy as np
import pandas as pd
import networkx as nx
import topology
import electrostatic

class model_class:

    def __init__(self, net_topology : np.array, inv_capacitance_matrix : np.array, tunnel_order=1)->None:
        """
        Based on network topology new indices corresponding to each junction are defined
        net_topology            :   Network topology from topology_class
        inv_capacitance_matrix  :   Inverse capacitance matrix from electrostatic_class
        """
        
        self.tunnel_order = tunnel_order

        # CONST Parameter
        self.ele_charge = 0.160217662
        self.kb         = 1.38064852e-5

        # Topology Parameter
        self.N_particles            = net_topology.shape[0]
        self.N_junctions            = net_topology.shape[1]
        self.N_electrodes           = np.sum(net_topology[:,0] != -100)
        
        # Inverse Capacitance Matrix
        self.inv_capacitance_matrix = inv_capacitance_matrix
        
        if self.tunnel_order >= 1:

            # Connection Array to show which nps and electrodes are connected 
            connections = np.zeros((self.N_particles+self.N_electrodes, self.N_junctions))
            connections.fill(-100)
            connections[self.N_electrodes:,:]   = net_topology
            nth_e                               = 1
            nth_np                              = 0
            while ((nth_np < self.N_particles)) and (nth_e <= self.N_electrodes):
                if int(net_topology[nth_np,0]) == nth_e:
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
        Potential Vector with [U_e1,U_e2,...,U_eN,U_np1,U_np2,...,U_npN]
        voltage_values  :   [U_e1,U_e2,...,U_eN]
        """

        self.potential_vector                       = np.ones(self.N_electrodes+self.N_particles)
        self.potential_vector[0:self.N_electrodes]  = voltage_values[:-1]

    def init_const_capacitance_values(self)->None:
        """
        Init array containing C_ii + C_jj + C_ij s
        um for free energy calculation
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

        return self.potential_vector

    def return_const_capacitance_values(self)->np.array:

        return self.const_capacitance_values, self.const_capacitance_values_co1, self.const_capacitance_values_co2
    
    def return_particle_electrode_count(self)->float:

        return self.N_electrodes, self.N_particles
    
    def return_advanced_indices(self)->tuple:

        return self.adv_index_rows, self.adv_index_cols, self.co_adv_index1, self.co_adv_index2, self.co_adv_index3
    
    def return_const_temperatures(self, T=0.28)->np.array:

        return np.repeat(T*self.kb, len(self.adv_index_rows)), np.repeat(T*self.kb, len(self.co_adv_index1))

    def return_const_resistances(self, R=25e-12)->np.array:

        if self.tunnel_order >= 1:
            const_R     = np.repeat(R*self.ele_charge*self.ele_charge, len(self.adv_index_rows))
            const_R_co1 = np.array([])
            const_R_co2 = np.array([])

        if self.tunnel_order >= 2:
            const_R_co1 = np.repeat(R*self.ele_charge*self.ele_charge, len(self.co_adv_index2))
            const_R_co2 = np.repeat(R*self.ele_charge*self.ele_charge, len(self.co_adv_index3))

        return const_R, const_R_co1, const_R_co2
    
    def return_random_resistances(self, R=25e-12, Rstd=2e-12):

        if self.tunnel_order >= 1:
            const_R     = np.repeat(np.abs(np.random.normal(R*self.ele_charge*self.ele_charge, Rstd)), len(self.adv_index_rows))
            const_R_co1 = np.array([])
            const_R_co2 = np.array([])

        if self.tunnel_order >= 2:
            "Random resistance not supported in tunnel_order >= 2"

        return const_R, const_R_co1, const_R_co2
    
    def set_np_resistance(self, np_index, np_R, resistance_arr):
        
        resistance_arr[np.where(self.adv_index_cols==np_index)[0]] = np_R

        return resistance_arr

    def return_transition_df(self)->pd.DataFrame:
        
        df      = pd.DataFrame()
        df['i'] = self.adv_index_rows + self.co_adv_index1
        df['j'] = self.adv_index_cols + self.co_adv_index2

        if self.tunnel_order > 1:

            df['k'] = [-1]*len(self.adv_index_rows) + self.co_adv_index3

        return df 

if __name__ == "__main__":

    N = 3
    cubic_topology = topology.topology_class()
    cubic_topology.cubic_network(N_x=N, N_y=N, N_z=1)
    cubic_topology.set_electrodes_based_on_pos([[0,0,0], [int((N-1)/2),0,0], [N-1,0,0], [0,int((N-1)/2),0], [0,N-1,0], [N-1,int((N-1)/2),0], [int((N-1)/2),(N-1),0], [N-1,N-1,0]])
    cubic_net = cubic_topology.return_net_topology()

    print("Cubic Network Topology:\n", cubic_net)

    # Electrostatic
    cubic_electrostatic = electrostatic.electrostatic_class(net_topology=cubic_net)
    cubic_electrostatic.calc_capacitance_matrix()
    cubic_electrostatic.init_charge_vector(voltage_values=np.random.rand(9))

    # Model
    inv_capacitance_matrix  = cubic_electrostatic.return_inv_capacitance_matrix()
    charge_vector           = cubic_electrostatic.return_charge_vector()

    cubic_model = model_class(net_topology=cubic_net, inv_capacitance_matrix=inv_capacitance_matrix, tunnel_order=1)
    adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2, co_adv_index3 = cubic_model.return_advanced_indices()