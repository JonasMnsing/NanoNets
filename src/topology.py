# import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
from typing import List

class topology_class:
    """
    Class to setup nanoparticle network topology and connect electrodes. The class supports regular cubic
    grid and random topologies. Electrodes are connected to nanoparticle indices or nanopartciles closest
    to a fixed position.   

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
    G : nx.DiGraph
        NetworkX directed graph of the nanoparticle network
    pos : dict
        dictonary of nanoparticle (keys) positions (values) 
    net_topology : array
        Network topology. Rows represent nanoparticles. First column stores connected electrodes.
        Second to last columns store connected nanoparticles.
    
    Methods
    -------
        cubic_network(N_x : int, N_y : int, N_z : int)
            Setup a regular cubic grid of nanoparticles
        random_network(N_particles : int, N_junctions : int)
            Setup a random regular network of nanoparticles. The resulting network might be higher dimensional.
        attach_np_to_gate(self, gate_nps=None)
            Attach NPs to gate electrode. If gate_nps==None, all NPs are attached
        add_electrodes_to_random_net(electrode_positions : List[List])
            List which contains electrode positions [x,y]. Nanoparticles sit inside a box of lengths [1,1].
        graph_to_net_topology()
            Transfer directed graph to net_topology array
        set_electrodes_based_on_pos(particle_pos : List[List], N_x : int, N_y : int)
            Attach electrode to spcific nanoparticle positions.
            Only use this method for regular grid like networks
        set_electrodes_randomly(N_electrodes : int)
            Attache electrodes randomly to nanoparticles
        return_net_topology()
    """

    def __init__(self, seed=None)->None:
        """
        Parameters
        ----------
        seed : int
            Seed for random number generator
        """

        self.N_particles    = 0
        self.N_electrodes   = 0
        self.rng            = np.random.default_rng(seed=seed)

    def cubic_network(self, N_x : int, N_y : int, N_z : int)->None:
        """
        Setup a regular cubic grid of nanoparticles

        Parameters
        ----------
        N_x : int
            Number of nanoparticles in x-direction
        N_y : int
            Number of nanoparticles in y-direction
        N_z : int
            Number of nanoparticles in z-direction
        """

        self.N_particles    = N_x*N_y*N_z
        nano_particles_pos  = []
        n_NN                = 0

        # Assign coordinates to each nanoparticle
        for i in range(0,N_z):
            for j in range(0,N_y):
                for k in range(0,N_x):
                    nano_particles_pos.append([0, k, j, i])

        self.pos    = {i : pos[1:3] for i, pos in enumerate(nano_particles_pos)}
        self.G      = nx.DiGraph()
        self.G.add_nodes_from(np.arange(self.N_particles))

        # Define max number of next neigbors by network dimension
        if ((N_x > 1) and (N_y > 1) and (N_z > 1)):
            self.N_junctions = 6

        elif (((N_x > 1) and (N_y > 1)) or ((N_x > 1) and (N_z > 1)) or ((N_y > 1) and (N_z > 1))):
            self.N_junctions = 4

        else:
            self.N_junctions = 2

        # Fill network topology with placeholder (-100)
        self.net_topology = np.zeros(shape=(self.N_particles, self.N_junctions + 1))
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
                    self.G.add_edge(i,j)
                    self.G.add_edge(j,i)
                    
                if (n_NN == self.N_junctions):
                    break
                
            n_NN = 0
        
    def random_network(self, N_particles : int, N_junctions : int)->None:
        """
        Setup a random regular network of nanoparticles. The resulting network might be higher dimensional.

        Parameters
        ----------
        N_particles : int
            Number of nanoparticles
        N_junctions : int
            Number of junctions per nanoparticle
        """
        
        self.N_particles    = N_particles
        self.N_junctions    = N_junctions
        not_connected       = True
        
        # Generate random graphs until a graph is connected
        while not_connected:

            self.G          = nx.random_regular_graph(N_junctions, self.N_particles)
            not_connected   = not(nx.is_connected(self.G))

        # Make graph directed and position nanoparticles
        self.G   = self.G.to_directed()
        self.pos = nx.kamada_kawai_layout(self.G)
        self.pos = nx.spring_layout(self.G, pos=self.pos)
    
    def attach_np_to_gate(self, gate_nps=None)->None:
        """
        Attach NPs to gate electrode. If gate_nps==None, all NPs are attached
        ----------
        gate_nps : array
            Nanoparticles capacitively couple to gate
        """

        if gate_nps == None:
            self.gate_nps = np.ones(self.N_particles)
        else:
            self.gate_nps = gate_nps

    def add_electrodes_to_random_net(self, electrode_positions : List[List])->None:
        """
        Position electrodes to fixed spots (x,y) and attach them to closest nanoparticles

        Parameters
        ----------
        electrode_positions : list
            List which contains electrode positions [x,y]. Nanoparticles sit inside a box of lengths [1,1].
        """

        self.N_electrodes   = len(electrode_positions)
        rename              = {i:i+self.N_electrodes for i in range(len(self.G.nodes))}
        self.G              = nx.relabel_nodes(G=self.G, mapping=rename)
        self.pos            = {k+self.N_electrodes: v for k, v in self.pos.items()}

        node_positions  = pd.DataFrame(self.pos).T.sort_index()
        used_nodes      = []

        # For each electrode position
        for node, e_pos in enumerate(electrode_positions):
            
            # Compute distance between all nanoparticle and current electrode position
            node_positions['d'] = np.sqrt((e_pos[0]-node_positions[0])**2 + (e_pos[1]-node_positions[1])**2)
            
            # Find closest nanoparticle which is not already connected to an electrode
            for i in range(len(node_positions)):

                connected_node = node_positions.sort_values(by='d').index[i]
                if connected_node in used_nodes:
                    continue
                else:
                    used_nodes.append(connected_node)
                    break
            
            # Add resulting connection as edge to directed graph
            self.G.add_edge(node,connected_node)
            self.G.add_edge(connected_node,node)
            self.pos[node] = e_pos

    def graph_to_net_topology(self)->None:
        """
        Transfer directed graph to net_topology array
        """
    
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

    def set_electrodes_based_on_pos(self, particle_pos : List[List], N_x : int, N_y : int)->None:
        """
        Attach electrode to spcific nanoparticle positions.
        Only use this method for regular grid like networks

        Parameters
        ----------
        particle_pos : list
            List which contains particle positions [x,y,z], which will be connected to electrodes
        N_x : int
            Number of nanoparticles in x-direction
        N_y : int
            Number of nanoparticles in y-direction
        """

        self.N_electrodes = len(particle_pos)
        self.G.add_nodes_from(-np.arange(1,self.N_electrodes+1))

        for n, pos in enumerate(particle_pos):

            p = pos[1]*N_x + pos[0] + pos[2]*N_x*N_y
            self.net_topology[p,0] = 1 + n
            
            self.G.add_edge((-1-n),p)
            self.G.add_edge(p,(-1-n))

            if (pos[0] == 0):
                self.pos[-1-n] = (pos[0]-1,pos[1])
            elif (pos[0] == (N_x-1)):
                self.pos[-1-n] = (pos[0]+1,pos[1])
            elif (pos[1] == 0):
                self.pos[-1-n] = (pos[0],pos[1]-1)
            else:
                self.pos[-1-n] = (pos[0],pos[1]+1)

    def set_electrodes_randomly(self, N_electrodes : int)->None:
        """
        Attache electrodes randomly to nanoparticles
        
        Parameters
        ----------
        N_electrodes : int
            Number of electrodes
        """

        self.N_electrodes   = N_electrodes
        particles           = self.rng.choice(self.N_particles, size=self.N_electrodes, replace=False)

        for n,p in enumerate(particles):

            self.net_topology[p,0] = 1 + n

    def return_net_topology(self)->np.array:
        """
        Returns
        -------
        net_topology : array
            Network topology. Rows represent nanoparticles. First column stores connected electrodes.
            Second to last columns store connected nanoparticles.
        """

        return self.net_topology
    
    # def return_graph_object(self)->nx.Graph:
    #     """
    #     Returns
    #     -------
    #     G : nx.DiGraph
    #         NetworkX directed graph of the nanoparticle network
    #     """

    #     return self.G

###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Cubic Network Topology
    N_x, N_y, N_z   = 3,3,1
    electrode_pos   = [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]
    cubic_topology  = topology_class()
    cubic_topology.cubic_network(N_x, N_y, N_z)
    cubic_topology.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    cubic_net = cubic_topology.return_net_topology()

    print("Cubic Network Topology:\n", cubic_net)
    
    # Disordered Network Topology
    N_particles, N_junctions    = 20,4
    electrode_pos               = [[-1,-1],[1,-1],[-1,1],[1,1]]
    random_topology = topology_class()
    random_topology.random_network(N_particles, N_junctions)
    random_topology.add_electrodes_to_random_net(electrode_pos)
    random_topology.graph_to_net_topology()

    random_net = random_topology.return_net_topology()

    print("Disordered Network Topology:\n", random_net)