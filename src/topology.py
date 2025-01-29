import numpy as np
import networkx as nx
import pandas as pd
from typing import List
from scipy.spatial import Delaunay

class topology_class:
    """
    Class to set up nanoparticle network topology and connect electrodes.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles.
    N_electrodes : int
        Number of electrodes.
    N_junctions : int
        Number of junctions per nanoparticle.
    rng : numpy.random.Generator
        Random number generator instance.
    G : nx.DiGraph
        NetworkX directed graph of the nanoparticle network.
    pos : dict
        Dictionary of nanoparticle (keys) positions (values).
    net_topology : np.ndarray
        Network topology matrix.
    
    Methods
    -------
        cubic_network(N_x : int, N_y : int, N_z : int)
            Setup a regular cubic grid of nanoparticles
        random_network(N_particles : int, N_junctions : int)
            Setup a random regular network of nanoparticles. The resulting network might be higher dimensional.
        add_electrodes_to_random_net(electrode_positions : List[List])
            List which contains electrode positions [x,y]. Nanoparticles sit inside a box of lengths [1,1].
        graph_to_net_topology()
            Transfer directed graph to net_topology array
        set_electrodes_based_on_pos(particle_pos : List[List], N_x : int, N_y : int)
            Attach electrode to spcific nanoparticle positions.
            Only use this method for regular grid like networks
        return_net_topology()
    """

    NO_CONNECTION = -100 # Placeholder for unconnected junctions

    def __init__(self, electrode_type: List[str] = None, seed: int = None) -> None:
        """
        Parameters
        ----------
        electrode_type : List[str]
            List indicating the type of electrodes.
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        self.rng            = np.random.default_rng(seed)
        self.N_particles    = 0
        self.N_electrodes   = 0

        if electrode_type is not None:
            self.electrode_type = np.array(electrode_type)

    def cubic_network(self, N_x: int, N_y: int, N_z: int) -> None:
        """
        Define a cubic lattice of nanoparticles.

        Parameters
        ----------
        N_x : int
            Number of nanoparticles in x-direction (columns).
        N_y : int
            Number of nanoparticles in y-direction (rows).
        N_z : int
            Number of nanoparticles in z-direction (depth).
        """
        if N_x <= 0 or N_y <= 0 or N_z <= 0:
            raise ValueError("Dimensions N_x, N_y, N_z must be positive integers.")
        
        # Calculate total number of NPs
        self.N_particles    = N_x*N_y*N_z

        # Create nanoparticle positions
        nano_particles_pos  = []
        for z in range(0,N_z):
            for y in range(0,N_y):
                for x in range(0,N_x):
                    nano_particles_pos.append([0, x, y, z])

        # Store 2D positions for visualization
        self.pos    = {i : [pos[1],pos[2]] for i, pos in enumerate(nano_particles_pos)}
        
        # Create the directed graph
        self.G  = nx.DiGraph()
        self.G.add_nodes_from(np.arange(self.N_particles))

        # Determine the number of junctions (neighbors) based on dimensionality
        if ((N_x > 1) and (N_y > 1) and (N_z > 1)):
            self.N_junctions = 6
        elif (((N_x > 1) and (N_y > 1)) or ((N_x > 1) and (N_z > 1)) or ((N_y > 1) and (N_z > 1))):
            self.N_junctions = 4
        else:
            self.N_junctions = 2

        # Initialize network topology with placeholders (-100)
        self.net_topology = np.full((self.N_particles, self.N_junctions + 1), fill_value=self.NO_CONNECTION)

        # Populate the graph and net topology with neighbors
        for idx, pos1 in enumerate(nano_particles_pos):
            n_NN = 0  # Neighbor count for this nanoparticle

            for jdx, pos2 in enumerate(nano_particles_pos):

                # Calculate distance to identify direct neighbors
                distance = np.sqrt((pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2 + (pos1[3] - pos2[3])**2)

                if distance == 1:  # If neighbors

                    self.net_topology[idx, n_NN + 1] = jdx
                    self.G.add_edge(idx,jdx)
                    self.G.add_edge(jdx,idx)
                    n_NN += 1
                    
                if (n_NN == self.N_junctions):
                    break
         
    def random_network(self, N_particles: int, N_junctions: int = 0)->None:
        """Setup a random network of nanoparticles.

        Parameters
        ----------
        N_particles : int
            Number of nanoparticles
        N_junctions : int
            Number of junctions per nanoparticle in a random regular netwokr. If 0, a Delaunay triangulation is used
            and the network is forced to be planar (2D), by default 0.
        """
        
        self.N_particles    = N_particles
        self.N_junctions    = N_junctions

        # Apply validation only if N_junctions is non-zero (random regular network case)
        if N_junctions > 0 and (N_junctions >= N_particles):
            raise ValueError("N_junctions must be a positive integer less than N_particles.")

        if N_junctions == 0:
            # Node positions
            angles  = self.rng.uniform(0,2*np.pi, self.N_particles)
            radii   = np.sqrt(self.rng.uniform(0, 1, self.N_particles))
            pos     = [(r * np.cos(a), r * np.sin(a)) for r, a in zip(radii, angles)]
            self.G  = nx.Graph()

            # Add nodes to graph object
            for i, p in enumerate(pos):
                self.G.add_node(i, pos=p)

            # Delaunay Triangulation
            tri     = Delaunay(pos)
            edges   = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                    edges.add(edge)
            
            edges = list(edges)

            # Add edges and uptade pos
            self.G.add_edges_from(edges)
            self.pos = {i : p for i, p in enumerate(pos)}
            self.N_junctions = np.max([val for (node, val) in self.G.degree()])

        else:
            # Random regular graph generation until connected
            while True:

                self.G  = nx.random_regular_graph(N_junctions, self.N_particles)
                if nx.is_connected(self.G):
                    break

            # Make graph directed and position nanoparticles
            self.G   = self.G.to_directed()
            self.pos = nx.kamada_kawai_layout(self.G)
            self.pos = nx.spring_layout(self.G, pos=self.pos)

    def set_electrodes_based_on_pos(self, particle_pos: List[List[int]], N_x: int, N_y: int)->None:
        """Attach electrode to spcific nanoparticle positions. Only use this method for cubic lattice networks

        Parameters
        ----------
        particle_pos : List[List[int]]
            List of particle positions [x,y,z], where each particle will be connected to an electrode.
        N_x : int
            Number of nanoparticles in x-direction.
        N_y : int
            Number of nanoparticles in y-direction.
        """
        self.N_electrodes = len(particle_pos)   # The number of electrodes equals the number of particles connected to electrodes.

        # Add the electrode nodes to the graph, using negative indices for electrodes
        electrode_nodes = -np.arange(1,self.N_electrodes+1)
        self.G.add_nodes_from(electrode_nodes)

        # Attach each electrode to its corresponding particle based on particle positions
        for n, pos in enumerate(particle_pos):
            # Convert 3D position (x, y, z) into a 1D index in the nanoparticle network
            p = pos[2]*N_x*N_y + pos[1]*N_x + pos[0]
            self.net_topology[p,0] = 1 + n  # Store the electrode number (starting from 1)
            
            # Connect the electrode node to the nanoparticle
            electrode_node  = electrode_nodes[n]
            self.G.add_edge(electrode_node,p)
            self.G.add_edge(p,electrode_node)

            # Set electrode positions, adjusting based on boundary conditions
            if (pos[0] == 0):           # If the particle is at the left boundary (x == 0)
                self.pos[electrode_node] = (pos[0]-1,pos[1])
            elif (pos[0] == (N_x-1)):   # If the particle is at the right boundary (x == N_x-1)
                self.pos[electrode_node] = (pos[0]+1,pos[1])
            elif (pos[1] == 0):         # If the particle is at the bottom boundary (y == 0)
                self.pos[electrode_node] = (pos[0],pos[1]-1)
            else:                       # Default case (otherwise move the electrode in the positive direction)
                self.pos[electrode_node] = (pos[0],pos[1]+1)

    def add_electrodes_to_random_net(self, electrode_positions : List[List[int]])->None:
        """Position electrodes to fixed spots (x,y) and attach them to the closest nanoparticles.

        For each electrode, the closest nanoparticle that is not already connected to an electrode is identified,
        and a connection is made.

        Parameters
        ----------
        electrode_positions : List[List[int]]
            List of electrode positions [x, y]. Nanoparticles sit inside a box of lengths [-1, 1].
        """
        # Number of electrodes
        self.N_electrodes   = len(electrode_positions)

        # Convert nanoparticle positions to a DataFrame for easy distance calculations
        node_positions  = pd.DataFrame(self.pos).T.sort_index()
        used_nodes      = []    # List to track already used nanoparticles

        # Loop through each electrode position
        for node, e_pos in enumerate(electrode_positions):
            # Compute the distance between each nanoparticle and the current electrode position
            node_positions['d'] = np.sqrt((e_pos[0]-node_positions[0])**2 + (e_pos[1]-node_positions[1])**2)
            
            # Sort nanoparticle positions by distance and find the closest unused nanoparticle
            for i in node_positions.sort_values(by='d').index:
                if i not in used_nodes:
                    # Add the nanoparticle to the used set and break out of the loop
                    used_nodes.append(i)
                    closest_nanoparticle = i
                    break
            
            # Create a bidirectional edge between the electrode and the selected nanoparticle
            self.G.add_edge(-node-1,closest_nanoparticle)
            self.G.add_edge(closest_nanoparticle,-node-1)
            
            # Store the electrode position
            self.pos[-node-1] = e_pos
    
    def add_np_to_e_pos(self):
        """Attach nanoparticles to all floating electrodes and update the network topology accordingly.
        """
        # Find indices of floating electrodes
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        
        # Increase the number of nanoparticles based on the number of floating electrodes
        self.N_particles    += len(floating_electrodes)

        # Loop through each floating electrode
        for electrode_index in floating_electrodes:
            # Find the nanoparticle that is connected to the floating electrode
            adj_np  = np.where(self.net_topology[:,0]==(electrode_index+1))[0][0]
            
            # Create a new row for the new nanoparticle and set the connections
            new_nn      = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
            new_nn[0]   = electrode_index+1                                         # First column: connect to the electrode   
            new_nn[1]   = adj_np                                                    # Second column: connect to the adjacent nanoparticle

            # Add the new nanoparticle and its connections to the network topology
            self.net_topology           = np.vstack((self.net_topology,new_nn))

            # Update the adjacent nanoparticle's connection to remove the floating electrode
            first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
            self.net_topology[adj_np,first_free_spot]   = self.net_topology.shape[0]-1 
            self.net_topology[adj_np,0]                 = self.NO_CONNECTION

               
    def graph_to_net_topology(self)->None:
        """Transfer directed graph to net_topology array.
    
        This function creates a network topology matrix where:
        - The first column represents the connected electrodes.
        - The subsequent columns represent neighboring nanoparticles.
        """
    
        # Initialize net_topology with the "no connection" placeholder value
        net_topology = np.full(shape=(self.N_particles, self.N_junctions + 1), fill_value=self.NO_CONNECTION)

        # Iterate through each nanoparticle node in the graph
        for node in range(0,len(self.G.nodes)-self.N_electrodes):
            for i, neighbor in enumerate(self.G.neighbors(node)):
                if neighbor >= 0:
                    net_topology[node,i+1] = neighbor
                else:
                    net_topology[node,0] = -neighbor
                    i -= 1

        # Store the generated net topology in the class attribute
        self.net_topology = net_topology

    def return_net_topology(self)->np.array:
        """
        Returns the network topology matrix.
        
        The network topology is structured as follows:
        - Rows represent nanoparticles.
        - The first column stores connected electrodes.
        - The second to last columns store connected nanoparticles.

        Returns
        -------
        net_topology : np.ndarray
            Network topology matrix.
        """
        return self.net_topology
    
    def __str__(self):
        return f"Topology Class with {self.N_particles} particles, {self.N_junctions} junctions.\nNetwork Topology:\n{self.net_topology}"

    # Delete this method
    # def attach_np_to_gate(self, gate_nps=None)->None:
    #     """ Attach NPs to gate electrode. If gate_nps==None, all NPs are attached
        
    #     Parameters
    #     ----------
    #     gate_nps : array
    #         Nanoparticles capacitively coupled to gate
    #     """

    #     if gate_nps == None:
    #         self.gate_nps = np.ones(self.N_particles)
    #     else:
    #         self.gate_nps = gate_nps
    
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Cubic Network Topology
    N_x, N_y, N_z   = 3,3,1
    electrode_pos   = [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]
    cubic_topology  = topology_class()
    cubic_topology.cubic_network(N_x, N_y, N_z)
    cubic_topology.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    print(cubic_topology)
    
    # Disordered Network Topology
    N_particles, N_junctions    = 20,0
    electrode_pos               = [[-1,-1],[1,-1],[-1,1],[1,1]]
    random_topology             = topology_class()
    random_topology.random_network(N_particles, N_junctions)
    random_topology.add_electrodes_to_random_net(electrode_pos)
    random_topology.graph_to_net_topology()
    print(random_topology)
