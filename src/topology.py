import numpy as np
from typing import List

class topology_class:
    """
    Class to setup network topology and connect electrodes. Cubic and random topologies are supported.
    Electrodes may either be connected to specific nanoparticle positions or randomly.

    net_topology    :   np.array representing nanoparticles as rows and connections as columns
                        -100 as element indicates no connection
                        First column corresponds to connected electrode
                        Second to last column corresponds to connected nanoparticles
                        Number of columns indicates maximum number of conneections per nanoparticle
    """

    def __init__(self, seed=None) -> None:
        """
        Init a random number generator needed for random network topologies
        """

        self.rng = np.random.default_rng(seed=seed)

    def cubic_network(self, N_x : int, N_y : int, N_z : int) -> None:
        """
        Init cubic packed network of nanoparticles
        N_x         :   Number of nps in x direction
        N_y         :   Number of nps in y direction
        N_z         :   Number of nps in z direction
        N_particles :   Number of nps
        """

        self.N_x            = N_x           # Number of nps in x direction
        self.N_y            = N_y           # Number of nps in y direction
        self.N_z            = N_z           # Number of nps in z direction
        self.N_particles    = N_x*N_y*N_z   # Number of particles
        nano_particles_pos  = []            # Storage for particle coordinates
        n_NN                = 0             # Number of next neighbor for a specific NP

        # Assign coordinates to each nanoparticle
        for i in range(0,self.N_z):
            for j in range(0,self.N_y):
                for k in range(0,self.N_x):
                    nano_particles_pos.append([0, k, j, i])

        # Define max number of next neigbors based on network dimension
        if ((self.N_x > 1) and (self.N_y > 1) and (self.N_z > 1)):
            max_NN = 6

        elif (((self.N_x > 1) and (self.N_y > 1)) or ((self.N_x > 1) and (self.N_z > 1)) or ((self.N_y > 1) and (self.N_z > 1))):
            max_NN = 4

        else:
            max_NN = 2

        # Network Topology filled with -100 (no connection)
        self.net_topology = np.zeros(shape=(self.N_particles, max_NN + 1))
        self.net_topology.fill(-100)

        # Attach neighbors to each nanoparticle
        for i in range(0, self.N_particles):

            x1 = nano_particles_pos[i][1]
            y1 = nano_particles_pos[i][2]
            z1 = nano_particles_pos[i][3]

            for j in range(0, self.N_particles):

                x2 = nano_particles_pos[j][1]
                y2 = nano_particles_pos[j][2]
                z2 = nano_particles_pos[j][3]

                distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

                # Check if particles are adjacent
                if (distance == 1):

                    self.net_topology[i, n_NN + 1] = j
                    n_NN = n_NN + 1
                    
                if (n_NN == max_NN):
                    break
                
            n_NN = 0

    def random_network(self, N_particles : int, N_junctions_lower : int, N_junctions_upper : int, p=0.1)->None:
        """
        !!!!!!!!!!!!!!!!!DOES NOT WORK PROPERLY!!!!!!!!!!!!!!!!!
        Init random packed network topology.
        For each np its number of next neighbors is either sampled from a binomial distribution or uniform distribution
        N_particles         :   Number of nps
        N_junctions_lower   :   Lower bound for number of junctions per np
        N_junctions_upper   :   Upper bound for number of junctions per np
        p                   :   Probability of success (binomial), if p = 0 use uniform_dist(N_junctions_lower,N_junctions_upper)
        """

        self.N_particles        = N_particles
        self.N_junctions_lower  = N_junctions_lower
        self.N_junctions_upper  = N_junctions_upper
        self.p                  = p

        if p == 0.0:
            junction_sample = range(N_junctions_lower,N_junctions_upper+1)

        neighbour_sample = range(0,self.N_particles)

        # FIRST: Sample Number of Junctions for each Nanoparticle
        #########################################################
        self.net_topology = np.zeros(shape=(self.N_particles, self.N_junctions_upper + 1))
        self.net_topology.fill(-200)

        for i_1 in range(0, self.N_particles):

            junction_not_in_range = True

            while (junction_not_in_range):
                
                if (p != 0.0):
                    N_junctions = self.rng.binomial(n=self.N_junctions_upper/p, p=p)
                else:
                    N_junctions = self.rng.choice(junction_sample)

                if ((N_junctions >= self.N_junctions_lower) and (N_junctions <= self.N_junctions_upper)):
                    junction_not_in_range = False

            for i_2 in range(0, N_junctions + 1):
                
                self.net_topology[i_1,i_2] = -100
        
        # SECOND: For each Junction attach a random Nanoparticle
        ########################################################
        for p1 in range(0, self.N_particles):

            current_neighbours = []

            for j1 in range(1, self.N_junctions_upper + 1):

                if (self.net_topology[p1,j1] == -100):

                    neighbour_not_found = True

                    while (neighbour_not_found):

                        # Sample a new neighbour
                        neighbour = self.rng.choice(neighbour_sample)
                        
                        current_n_size = len(current_neighbours)

                        if (current_n_size == (self.N_particles - 1)):
                            
                            self.random_network(self.N_particles, self.N_junctions_lower, self.N_junctions_upper, self.p)

                        # If new neighbour is already a neighbour or new neighbour equals the current particle i
                        while ((neighbour in current_neighbours) or (neighbour == p1)):

                            neighbour = self.rng.choice(neighbour_sample)
                            
                        # Go to neighbour index and loop through each junction
                        for j2 in range(1, self.N_junctions_upper + 1):

                            # If there is still a place left for an additional neighbour
                            if (self.net_topology[neighbour, j2] == -100):

                                # Attach particles
                                self.net_topology[neighbour, j2] = p1
                                self.net_topology[p1,j1]         = neighbour
                                neighbour_not_found              = False
                                break
                                
                        # Add neighbour to neighbour storage
                        current_neighbours.append(neighbour)
                        print(current_neighbours)
                        print(self.net_topology)
        
        self.net_topology[self.net_topology == -200] = -100

    # def connect_NP_random(N_NP, max_d=0.25, max_connection=4):

    #     not_connected   = True
    #     n_try           = 0

    #     while(not_connected):

    #         pos_e   = np.array([[-0.1,-0.1],
    #                             [0.5,-0.1],
    #                             [1.1,-0.1],
    #                             [-0.1,0.5],
    #                             [1.1,0.5],
    #                             [-0.1,1.1],
    #                             [0.5,1.1],
    #                             [1.1,1.1]])
            
    #         radius  = np.random.uniform(low=0, high=0.5, size=(N_NP,1))
    #         angle   = np.random.uniform(low=0, high=2, size=(N_NP,1))
    #         pos_o   = np.concatenate((radius, angle), axis=1)

    #         pos         = np.zeros(shape=(N_NP,2))
    #         pos[:,0]    = np.cos(pos_o[:,1]*np.pi)*pos_o[:,0]
    #         pos[:,1]    = np.sin(pos_o[:,1]*np.pi)*pos_o[:,0]
    #         pos         = pos + 0.5
    #         con_m       = np.zeros(shape=(N_NP+8,N_NP+8))

    #         for i in range(8):
    #             distance    = np.sqrt((pos_e[i][0]-pos[:,0])**2 + (pos_e[i][1]-pos[:,1])**2)
    #             j           = np.argmin(distance)+8
    #             con_m[i,j]  = 1
    #             con_m[j,i]  = 1

    #         for i in range(N_NP):
    #             n_cons = 0
    #             for j in range(i,N_NP):

    #                 distance = np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)

    #                 if distance < max_d:

    #                     con_m[i+8,j+8] = 1
    #                     con_m[j+8,i+8] = 1
    #                     n_cons += 1

    #                 if n_cons == max_connection:
    #                     break

    #         np.fill_diagonal(con_m,0)
    #         pos = np.concatenate((pos_e,pos))

    #         G = nx.DiGraph(con_m)
    #         not_connected = not(nx.is_strongly_connected(G))
    #         n_try += 1

    #         if n_try > 10000:
    #             print("Cannot find a connected Graph!")
    #             break

    #     return con_m, pos, G

    def set_electrodes_based_on_pos(self, particle_pos : List[List])->None:
        """
        Attach electrodes to nanoparticles
        Example:
        particle_pos    :   list of lists containg x,y,z position of np
                            example [[0,0,0],[4,2,1]] connects two electrodes
                            to np at x=0,y=0,z=0 and x=4,y=2,z=1
        """

        self.N_electrodes = len(particle_pos) 

        for n, pos in enumerate(particle_pos):

            p = pos[1]*self.N_x + pos[0] + pos[2]*self.N_x*self.N_y
            self.net_topology[p,0] = 1 + n

    def set_electrodes_randomly(self, N_electrodes : int)->None:
        """
        Attache electrodes randomly to nanoparticles
        N_electrodes    :   Number of electrodes to be attached
        """

        self.N_electrodes   = N_electrodes
        particles           = self.rng.choice(self.N_particles, size=self.N_electrodes, replace=False)

        for n,p in enumerate(particles):

            self.net_topology[p,0] = 1 + n

    def return_net_topology(self)->np.array:

        return self.net_topology

###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Cubic Network Topology
    cubic_topology = topology_class()
    cubic_topology.cubic_network(N_x=3, N_y=3, N_z=1)
    cubic_topology.set_electrodes_based_on_pos([[0,0,0],[2,2,0]])
    cubic_net = cubic_topology.return_net_topology()

    print("Cubic Network Topology:\n", cubic_net)

    # Disordered Network Topology
    # random_topology = topology_class()
    # random_topology.random_network(N_particles=20, N_junctions_lower=1, N_junctions_upper=2, p=0.0)
    # random_topology = random_topology.return_net_topology()

    # print("Disordered Network Topology:\n", random_topology)
