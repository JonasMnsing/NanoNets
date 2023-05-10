import numpy as np
import topology

class electrostatic_class:
    
    def __init__(self, net_topology : np.array, gate_nps=None, seed=None)->None:
        
        self.rng            = np.random.default_rng(seed=seed)
        self.net_topology   = net_topology
        self.N_particles    = self.net_topology.shape[0]
        self.N_junctions    = self.net_topology.shape[1]
        self.N_electrodes   = np.sum(self.net_topology[:,0] != -100)

        if gate_nps == None:
            self.gate_nps = np.ones(self.N_particles)
        else:
            self.gate_nps = gate_nps

    def delete_n_junctions(self, n : int):
        
        for i in range(n):
            
            not_found   = True
            runs        = 0

            while (not_found == True):

                np1 = np.random.randint(0, self.N_particles)
                np2 = np.random.randint(1, self.N_junctions)

                if (self.net_topology[np1,np2] != -100) and (self.net_topology[np1,0] == -100):
                    not_found = False
                
                runs += 1

                if (runs > 5000):
                    print("No Junction found!!!")
                    return
            
            np1_2 = int(self.net_topology[np1,np2])
            np2_2 = np.where(self.net_topology[np1_2,1:]==np1)[0][0]
            self.net_topology[np1,np2]          = -100
            self.net_topology[np1_2,np2_2+1]    = -100

    def mutal_capacitance_adjacent_spheres(self, eps_r : float, np_radius1 : float, np_radius2 : float, np_distance : float)->float:

        factor  = 4*3.14159265359*8.85418781762039*0.001*eps_r
        cap     = factor*((np_radius1*np_radius2)/(np_radius1 + np_radius2 + np_distance))

        return cap

    def self_capacitance_sphere(self, eps_s : float, np_radius : float)->float:

        factor  = 4*3.14159265359*8.85418781762039*0.001*eps_s
        cap     = factor*np_radius

        return cap
    
    def calc_capacitance_matrix(self, eps_r=2.6, eps_s=3.9, mean_radius=10.0, std_radius=0.0, np_distance=1.0, mean_radius2=10.0, std_radius2=0.0)->None:

        # Calculate Capacitance Values
        self.radius_vals  = np.abs(np.random.normal(loc=mean_radius, scale=std_radius, size=self.N_particles))
        
        # Two NP Types?
        if mean_radius2 != mean_radius:
            self.radius_vals[::2] = np.abs(np.random.normal(loc=mean_radius2, scale=std_radius2, size=self.N_particles))[::2]

        self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
        self.eps_r          = eps_r
        self.eps_s          = eps_s
        self.np_distance    = np_distance
        C_sum               = 0.0

        # Fill Capacitance Matrix based on Net Topology
        for i in range(self.N_particles):
            for j in range(self.N_junctions):
            
                neighbor = self.net_topology[i,j]

                if (neighbor != (-100)):

                    if (j == 0):

                        C_sum += self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[i], np_distance) # self.C_lead[i,j] 

                    else:
                        val = self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[int(neighbor)], np_distance)
                        self.capacitance_matrix[i,int(self.net_topology[i,j])] = -val
                        C_sum += val # self.C_node[i,j]

            # If Nanoparticle is affacted by Gate
            if (self.gate_nps[i] == 1):
                C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i]) #self.C_self[i,i]
            
            # Add total Capacitance to diagonal component
            self.capacitance_matrix[i,i] = C_sum
            C_sum = 0.0
        
        self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)

    def init_charge_vector(self, voltage_values : np.array)->None:
        """
        voltage_values  :   Array of voltages with [V_e1, V_e2, V_e3, ... V_G]
        """

        assert len(voltage_values) == self.N_electrodes + 1, "voltages_values has to have the same length as N_electrodes + 1"

        self.charge_vector = np.zeros(self.N_particles)

        for i in range(self.N_particles):

            if (self.net_topology[i,0] == -100):

                if self.gate_nps[i] == 1:
                    C_self  = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                    self.charge_vector[i] = voltage_values[-1]*C_self

                else:

                    self.charge_vector[i] = 0.0

            else:

                electrode_index = int(self.net_topology[i,0] - 1)
                
                if self.gate_nps[i] == 1:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.radius_vals[i], self.np_distance)
                    C_self  = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                    self.charge_vector[i] = voltage_values[electrode_index]*C_lead + voltage_values[-1]*C_self
                else:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.radius_vals[i], self.np_distance)
                    self.charge_vector[i] = voltage_values[electrode_index]*C_lead

    def get_charge_vector_offset(self, voltage_values : np.array)->np.array:
        """
        voltage_values  :   Array of voltages with [V_e1, V_e2, V_e3, ... V_G]
        """

        assert len(voltage_values) == self.N_electrodes + 1, "voltages_values has to have the same length as N_electrodes + 1"

        offset = np.zeros(self.N_particles)

        for i in range(self.N_particles):

            if (self.net_topology[i,0] == -100):

                if self.gate_nps[i] == 1:
                    C_self  = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                    offset[i] = voltage_values[-1]*C_self

                else:

                    offset[i] = 0.0

            else:

                electrode_index = int(self.net_topology[i,0] - 1)
                
                if self.gate_nps[i] == 1:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.radius_vals[i], self.np_distance)
                    C_self  = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                    offset[i] = voltage_values[electrode_index]*C_lead + voltage_values[-1]*C_self
                else:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.radius_vals[i], self.np_distance)
                    offset[i] = voltage_values[electrode_index]*C_lead

        return offset

    def return_charge_vector(self)->np.array:
        
        return self.charge_vector
    
    def return_capacitance_matrix(self)->np.array:
        
        return self.capacitance_matrix
        

    def return_inv_capacitance_matrix(self)->np.array:
        
        return self.inv_capacitance_matrix


###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Cubic Network Topology
    cubic_topology = topology.topology_class()
    cubic_topology.cubic_network(N_x=4, N_y=4, N_z=1)
    cubic_topology.set_electrodes_based_on_pos([[0,0,0],[3,0,0],[0,3,0],[3,3,0]])
    cubic_net = cubic_topology.return_net_topology()

    # Electrostatic
    cubic_electrostatic = electrostatic_class(net_topology=cubic_net)
    print(cubic_electrostatic.net_topology, "\n")
    cubic_electrostatic.delete_n_junctions(3)
    print(cubic_electrostatic.net_topology)
    cubic_electrostatic.calc_capacitance_matrix()
    cubic_electrostatic.init_charge_vector(voltage_values=np.array([0.05,0.2,0.3,0.1,0.0]))

    capacitance_matrix      = cubic_electrostatic.return_capacitance_matrix()
    inv_capacitance_matrix  = cubic_electrostatic.return_inv_capacitance_matrix()
    charge_vector           = cubic_electrostatic.return_charge_vector()

    print("Capacitance Matrix:\n", capacitance_matrix)
    print("Inverse Capacitance Matrix:\n", inv_capacitance_matrix)
    print("Initial Charge Vector:\n", charge_vector)



