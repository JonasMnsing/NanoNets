import numpy as np
import topology

class electrostatic_class:
    """
    Class to setup electrostatic properties of the nanoparticle network.
    This class depends on the topology_class.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of Junctions per Nanoparticle
    rng : Generator
        Bit Generator 
    net_topology : array
        Network topology. Rows represent nanoparticles. First column stores connected electrodes.
        Second to last columns store connected nanoparticles.
    gate_nps : array
        Nanoparticles capacitively couple to gate
    eps_r : float
        Permittivity of insulating material in between nanoparticles
    eps_s : float
        Permittivity of insulating enverionment, oxide layer
    np_distance : float
        Spacing in between nanoparticles (edge to edge)
    radius_vals : array
        Array including nanoparticle radii
    capacitance_matrix : array
        Array containing network capacitance values
    inv_capacitance_matrix : array
        Inverse of capacitance matrix
    charge_vector : array
        Charge values for each nanoparticle 

    Methods
    -------
    delete_n_junctions(n : int)
        Delete n junctions in network at random
    mutal_capacitance_adjacent_spheres(eps_r : float, np_radius1 : float, np_radius2 : float, np_distance : float)
        Calculate capacitance in between a spherical conductors - insulator - spherical conductors setup
    self_capacitance_sphere(eps_s : float, np_radius : float)
        Calculate self capacitance of a sphere in an insulating enverionment
    init_nanoparticle_radius(mean_radius=10.0, std_radius=0.0)
        Sample radii for all nanoparticles from |Gaus(mean_radius,std_radius)|
    update_nanoparticle_radius(nanoparticles : list, mean_radius=10.0, std_radius=0.0)
        Sample new radii for given nanoparticles from |Gaus(mean_radius,std_radius)|
    calc_capacitance_matrix(eps_r=2.6, eps_s=3.9, np_distance=1.0)
        Setup capacitance matrix.
    init_charge_vector(voltage_values : np.array)
        Initialize offset of charges put into the system by electrodes
    get_charge_vector_offset(voltage_values : np.array)
        Return offset of charges put into the system by electrodes
    return_charge_vector()
    return_capacitance_matrix()
    return_inv_capacitance_matrix()
    """
    
    def __init__(self, net_topology : np.array, gate_nps=None, seed=None)->None:
        """
        Parameters
        ----------
        net_topology : array
            Network topology. Rows represent nanoparticles. First column stores connected electrodes.
            Second to last columns store connected nanoparticles.
        gate_nps : array
            Nanoparticles capacitively couple to gate
        seed : int
            Seed for random number generator
        """

        self.rng            = np.random.default_rng(seed=seed)
        self.net_topology   = net_topology
        self.N_particles    = self.net_topology.shape[0]
        self.N_junctions    = self.net_topology.shape[1]
        self.N_electrodes   = np.sum(self.net_topology[:,0] != -100)

        if gate_nps == None:
            self.gate_nps = np.ones(self.N_particles)
        else:
            self.gate_nps = gate_nps

    def delete_n_junctions(self, n : int)->None:
        """
        Delete n junctions in network at random
        !!! Auch besser via NetworkX und is_connected lösen !!!

        Parameters
        ----------
        n : int
            Number of junctions to delete
        """

        for i in range(n):
            
            not_found   = True
            runs        = 0

            # Find a junction to be deleted
            while (not_found == True):

                np1 = np.random.randint(0, self.N_particles)
                np2 = np.random.randint(1, self.N_junctions)

                # Select to nanoparticles at random which are connected and not connected to an electrode
                if (self.net_topology[np1,np2] != -100) and (self.net_topology[np1,0] == -100):
                    not_found = False
                
                runs += 1

                if (runs > 5000):
                    print("No Junction found!!!")
                    return
            
            # Remove Junction
            np1_2 = int(self.net_topology[np1,np2])
            np2_2 = np.where(self.net_topology[np1_2,1:]==np1)[0][0]
            self.net_topology[np1,np2]          = -100
            self.net_topology[np1_2,np2_2+1]    = -100

    def mutal_capacitance_adjacent_spheres(self, eps_r : float, np_radius1 : float, np_radius2 : float, np_distance : float)->float:
        """
        Calculate capacitance in between a spherical conductors - insulator - spherical conductors setup

        Parameters
        ----------
        eps_r : float
            Permittivity of insulating material in between
        np_radius1 : float
            Radius first sphere (nanoparticle)
        np_radius2 : float
            Radius second sphere (nanoparticle)
        np_distance : float
            Spacing in between both spheres (edge to edge) 

        Returns
        -------
        cap : float
            capacitance value
        """

        factor  = 4*3.14159265359*8.85418781762039*0.001*eps_r
        cap     = factor*((np_radius1*np_radius2)/(np_radius1 + np_radius2 + np_distance))

        return cap

    def self_capacitance_sphere(self, eps_s : float, np_radius : float)->float:
        """
        Calculate self capacitance of a sphere in an insulating enverionment

        Parameters
        ----------
        eps_s : float
            Permittivity of insulating enverionment
        np_radius : float
            Radius sphere (nanoparticle)
        
        Returns
        -------
        cap : float
            capacitance value
        """

        factor  = 4*3.14159265359*8.85418781762039*0.001*eps_s
        cap     = factor*np_radius

        return cap
        
    def init_nanoparticle_radius(self, mean_radius=10.0, std_radius=0.0)->None:
        """
        Sample radii for all nanoparticles from |Gaus(mean_radius,std_radius)|

        Parameters
        ----------
        mean_radius : float
            Average nanoparticle radius
        std_radius : float
            Radius standard deviation
        """

        self.radius_vals  = np.abs(np.random.normal(loc=mean_radius, scale=std_radius, size=self.N_particles))

    def update_nanoparticle_radius(self, nanoparticles : list, mean_radius=10.0, std_radius=0.0)->None:
        """
        Sample new radii for given nanoparticles from |Gaus(mean_radius,std_radius)|

        Parameters
        ----------
        nanoparticles : list
            Nanoparticle indices
        mean_radius : float
            Average nanoparticle radius
        std_radius : float
            Radius standard deviation
        """

        self.radius_vals[nanoparticles] = np.abs(np.random.normal(loc=mean_radius, scale=std_radius, size=len(nanoparticles)))
        
    def calc_capacitance_matrix(self, eps_r=2.6, eps_s=3.9, np_distance=1.0)->None:
        """
        Setup capacitance matrix.

        Parameters
        ----------
        eps_r : float
            Permittivity of insulating material in between nanoparticles
        eps_s : float
            Permittivity of insulating enverionment, oxide layer
        np_distance : float
            Spacing in between nanoparticles (edge to edge)
        """

        self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
        self.eps_r              = eps_r
        self.eps_s              = eps_s
        self.np_distance        = np_distance
        C_sum                   = 0.0

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
        
        # Get inverse matrix
        self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)

    def init_charge_vector(self, voltage_values : np.array)->None:
        """
        Initialize offset of charges put into the system by electrodes

        Parameters
        ----------
        voltage_values : array
           Electrode voltages with [V_e1, V_e2, V_e3, ... V_G]
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
        Return offset of charges put into the system by electrodes

        Parameters
        ----------
        voltage_values : array
           Electrode voltages with [V_e1, V_e2, V_e3, ... V_G]

        Returns
        -------
        offset : array
            Charge offset
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
        """
        Returns
        -------
        charge_vector : array
            Charge values for each nanoparticle
        """
                
        return self.charge_vector
    
    def return_capacitance_matrix(self)->np.array:
        """
        Returns
        -------
        capacitance_matrix : array
            Array containing network capacitance values
        """

        return self.capacitance_matrix
        
    def return_inv_capacitance_matrix(self)->np.array:
        """
        Returns
        -------
        inv_capacitance_matrix : array
            Inverse of capacitance matrix
        """

        return self.inv_capacitance_matrix

###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Parameter
    N_x, N_y, N_z       = 3,3,1
    electrode_pos       = [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]
    radius, radius_std  = 10.0, 0.0
    eps_r, eps_s        = 2.6, 3.9
    np_distance         = 1
    voltage_values      = [0.1,0.2,-0.1,0.3,-0.8]

    # Cubic Network Topology
    cubic_topology  = topology.topology_class()
    cubic_topology.cubic_network(N_x, N_y, N_z)
    cubic_topology.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    cubic_net = cubic_topology.return_net_topology()

    print("Cubic Network Topology:\n", cubic_net)

    # Electrostatic
    cubic_electrostatic = electrostatic_class(net_topology=cubic_net)
    cubic_electrostatic.init_nanoparticle_radius(radius, radius_std)
    cubic_electrostatic.calc_capacitance_matrix(eps_r, eps_s, np_distance)
    cubic_electrostatic.init_charge_vector(voltage_values)

    capacitance_matrix      = cubic_electrostatic.return_capacitance_matrix()
    inv_capacitance_matrix  = cubic_electrostatic.return_inv_capacitance_matrix()
    charge_vector           = cubic_electrostatic.return_charge_vector()

    print("Capacitance Matrix:\n", capacitance_matrix)
    print("Inverse Capacitance Matrix:\n", inv_capacitance_matrix)
    print("Initial Charge Vector:\n", charge_vector)



