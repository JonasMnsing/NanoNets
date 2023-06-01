import numpy as np
import networkx as nx
import pandas as pd
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
        self.N_particles    = 0
        self.N_electrodes   = 0
        self.rng            = np.random.default_rng(seed=seed)

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

    def random_network(self, N_particles : int, N_junctions : int):
        """
        N_particles     :   Number of nodes in Graph
        N_junctions     :   Degree of each node
        Init a random regular graph of nanoparticles
        Nanoparticles are positioned based on force-directed graph drawing using Fruchterman-Reingold
        Thus, there topology will represent a local minimum.
        The resulting graph will not be planar (2D)!!! 
        """
        
        self.N_particles    = N_particles
        self.N_junctions    = N_junctions
        not_connected       = True
        
        while not_connected:

            self.G          = nx.random_regular_graph(N_junctions, self.N_particles)
            not_connected   = not(nx.is_connected(self.G))

        self.G   = self.G.to_directed()
        self.pos = nx.kamada_kawai_layout(self.G)
        self.pos = nx.spring_layout(self.G, pos=self.pos)

    def add_electrodes_to_random_net(self, electrode_positions : list):

        self.N_electrodes   = len(electrode_positions)
        rename              = {i:i+self.N_electrodes for i in range(len(self.G.nodes))}
        self.G              = nx.relabel_nodes(G=self.G, mapping=rename)
        self.pos            = {k+self.N_electrodes: v for k, v in self.pos.items()}

        node_positions  = pd.DataFrame(self.pos).T.sort_index()
        used_nodes      = []

        for node, e_pos in enumerate(electrode_positions):
            
            node_positions['d'] = np.sqrt((e_pos[0]-node_positions[0])**2 + (e_pos[1]-node_positions[1])**2)
            
            for i in range(len(node_positions)):

                connected_node = node_positions.sort_values(by='d').index[i]
                if connected_node in used_nodes:
                    continue
                else:
                    used_nodes.append(connected_node)
                    break
            
            self.G.add_edge(node,connected_node)
            self.G.add_edge(connected_node,node)
            self.pos[node] = e_pos

    def graph_to_net_topology(self):
    
        net_topology = np.zeros(shape=(self.N_particles,self.N_junctions+1))
        net_topology.fill(-100)

        for node in range(self.N_electrodes,len(self.G.nodes)):
            for i, neighbor in enumerate(self.G.neighbors(node)):
                if neighbor >= self.N_electrodes:
                    net_topology[node-self.N_electrodes,i+1] = neighbor-self.N_electrodes
                else:
                    net_topology[node-self.N_electrodes,0] = neighbor+1
                    i -= 1

        self.net_topology = net_topology

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
    random_topology = topology_class()
    random_topology.random_network(N_particles=20, N_junctions=4)
    random_topology.add_electrodes_to_random_net(electrode_positions=[[-1,-1]])
    random_topology.graph_to_net_topology()

    random_net = random_topology.return_net_topology()

    print("Disordered Network Topology:\n", random_net)
