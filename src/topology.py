import numpy as np
import networkx as nx
import pandas as pd
from typing import List, Optional
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
    N_x : int
        Number of nanoparticles in x-direction (only set for cubic networks).
    N_y : int
        Number of nanoparticles in y-direction (only set for cubic networks).
    N_z : int
        Number of nanoparticles in z-direction (only set for cubic networks).
    rng : numpy.random.Generator
        Random number generator instance.
    G : nx.DiGraph
        NetworkX directed graph of the nanoparticle network.
    pos : dict
        Dictionary mapping node indices to their 2D positions (x,y). For cubic networks,
        coordinates are integer grid positions. For random networks, coordinates are
        normalized to [-1,1] range. Electrode positions are placed relative to their
        connected nanoparticles.
    net_topology : np.ndarray
        Network topology matrix where each row represents a nanoparticle.
        First column indicates connected electrode (if any),
        subsequent columns indicate connections to other nanoparticles.
    
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
        export_network(filepath : str)
            Export the network configuration to a file.
        import_network(filepath : str)
            Import a network configuration from a file.
    """

    NO_CONNECTION = -100 # Placeholder for unconnected junctions

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default None.
        """
        self.rng            = np.random.default_rng(seed)
        self.N_particles    = 0
        self.N_electrodes   = 0
        self.G              = nx.DiGraph()
        self.pos            = {}
        self.net_topology   = np.array([])

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
        
        self.lattice = True

        self.N_x    = N_x
        self.N_y    = N_y
        self.N_z    = N_z
        
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
            self.N_junctions = 6+1
        elif (((N_x > 1) and (N_y > 1)) or ((N_x > 1) and (N_z > 1)) or ((N_y > 1) and (N_z > 1))):
            self.N_junctions = 4+1
        else:
            self.N_junctions = 2+1

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
            Number of junctions per nanoparticle in a random regular network. If 0, a Delaunay triangulation is used
            and the network is forced to be planar (2D), by default 0.
        """
        
        self.lattice        = False
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
            temp_G  = nx.Graph()

            # Add nodes to graph object
            for i, p in enumerate(pos):
                temp_G.add_node(i, pos=p)

            # Delaunay Triangulation
            tri     = Delaunay(pos)
            edges   = set()
            for simplex in tri.simplices:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i+1) % 3]]))
                    edges.add(edge)
            
            # Add edges to temporary undirected graph
            temp_G.add_edges_from(edges)
            
            # Convert to directed graph
            self.G = nx.DiGraph(temp_G)
            self.pos = {i : p for i, p in enumerate(pos)}
            self.N_junctions = np.max([val for (node, val) in temp_G.degree()])

        else:
            # Random regular graph generation until connected
            while True:
                temp_G = nx.random_regular_graph(N_junctions, self.N_particles)
                if nx.is_connected(temp_G):
                    break

            # Convert to directed graph
            self.G = nx.DiGraph(temp_G)
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
    
    def add_np_to_output(self):
        """Attach one nanoparticle to the output electrode (last electrode index).

        This method:
        1. Disconnects the existing nanoparticle from the output electrode
        2. Creates a new nanoparticle
        3. Connects the new nanoparticle between the previously connected nanoparticle and the output electrode
        """
        
        # Index of output electrode
        output_electrode_idx    = self.N_electrodes-1
        
        # Increase the number of nanoparticles by two
        prev_particle_count =   self.N_particles
        self.N_particles    +=  1

        # Find the nanoparticle that is connected to the floating electrode
        adj_np  = np.where(self.net_topology[:,0]==(output_electrode_idx+1))[0][0]

        # Create a new row for the first nanoparticle and set update connection
        new_nn_1    = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
        new_nn_1[0] = output_electrode_idx+1                                    # Connect to the electrode   
        new_nn_1[1] = adj_np                                                    # Connect to the adjacent nanoparticle

        # Add the new nanoparticles and their connections to the network topology
        self.net_topology   = np.vstack((self.net_topology,new_nn_1))

        # Update the adjacent nanoparticle's connection
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,first_free_spot]   = self.net_topology.shape[0]-1
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,0]                 = self.NO_CONNECTION

        # Remove old electrode connection
        self.G.remove_edge(adj_np,-output_electrode_idx-1)
        self.G.remove_edge(-output_electrode_idx-1,adj_np)

        # Add new nanoparticles
        self.G.add_node(prev_particle_count)

        # Add new edges
        self.G.add_edge(prev_particle_count,adj_np)
        self.G.add_edge(adj_np,prev_particle_count)
        self.G.add_edge(prev_particle_count,-output_electrode_idx-1)
        self.G.add_edge(-output_electrode_idx-1,prev_particle_count)

        # Update node positions
        x, y                            = self.pos[-output_electrode_idx-1]
        self.pos[prev_particle_count]   = (x,y)
        if self.lattice:
            if x == self.N_x:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0]+1,self.pos[-output_electrode_idx-1][1])
            elif x == -1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0]-1,self.pos[-output_electrode_idx-1][1])
            elif y == self.N_y:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0],self.pos[-output_electrode_idx-1][1]+1)
            elif y == -1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0],self.pos[-output_electrode_idx-1][1]-1)
        else:
            if x == 1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0]+1,self.pos[-output_electrode_idx-1][1])
            elif x == -1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0]-1,self.pos[-output_electrode_idx-1][1])
            elif y == 1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0],self.pos[-output_electrode_idx-1][1]+1)
            elif y == -1:
                self.pos[-output_electrode_idx-1]   = (self.pos[-output_electrode_idx-1][0],self.pos[-output_electrode_idx-1][1]-1)


    def add_two_in_parallel_np_to_output(self):
        """Attach two nanoparticles in parallel to the output electrode (last electrode index).

        This method:
        1. Disconnects the existing nanoparticle from the output electrode
        2. Creates two new nanoparticles
        3. Connects both new nanoparticles between the previously connected nanoparticle and the output electrode,
           forming a parallel configuration
        """
        
        # Index of output electrode
        output_electrode_idx    = self.N_electrodes-1
            
        # Increase the number of nanoparticles by two
        prev_particle_count =   self.N_particles
        self.N_particles    +=  2

        # Find the nanoparticle that is connected to the floating electrode
        adj_np  = np.where(self.net_topology[:,0]==(output_electrode_idx+1))[0][0]

        # Create a new row for the first nanoparticle and set update connection
        new_nn_1    = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
        new_nn_1[1] = adj_np                                                    # Connect to the adjacent nanoparticle
        new_nn_1[2] = self.N_particles-1                                        # Connect to second "new" nanoparticle

        # Create a new row for the first nanoparticle and set the connections
        new_nn_2    = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
        new_nn_2[0] = output_electrode_idx+1                                    # Connect to the electrode   
        new_nn_2[1] = adj_np                                                    # Connect to the adjacent nanoparticle
        new_nn_2[2] = self.N_particles-2                                        # Connect to first "new" nanoparticle

        # Add the new nanoparticles and their connections to the network topology
        self.net_topology   = np.vstack((self.net_topology,new_nn_1))
        self.net_topology   = np.vstack((self.net_topology,new_nn_2))

        # Update the adjacent nanoparticle's connection
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,first_free_spot]   = self.net_topology.shape[0]-2
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,first_free_spot]   = self.net_topology.shape[0]-1
        self.net_topology[adj_np,0]                 = self.NO_CONNECTION

        # Remove old electrode connection
        self.G.remove_edge(adj_np,-output_electrode_idx-1)
        self.G.remove_edge(-output_electrode_idx-1,adj_np)

        # Add new nanoparticles
        self.G.add_node(prev_particle_count)
        self.G.add_node(prev_particle_count+1)

        # Add new edges
        self.G.add_edge(prev_particle_count,adj_np)
        self.G.add_edge(adj_np,prev_particle_count)
        self.G.add_edge(prev_particle_count+1,adj_np)
        self.G.add_edge(adj_np,prev_particle_count+1)
        self.G.add_edge(prev_particle_count,prev_particle_count+1)
        self.G.add_edge(prev_particle_count+1,prev_particle_count)
        self.G.add_edge(prev_particle_count+1,-output_electrode_idx-1)
        self.G.add_edge(-output_electrode_idx-1,prev_particle_count+1)

    def add_two_in_series_np_to_output(self):
        """Attach two nanoparticles in series to the output electrode.

        This method:
        1. Disconnects the existing nanoparticle from the output electrode
        2. Creates two new nanoparticles
        3. Connects the new nanoparticles in series between the previously connected nanoparticle 
           and the output electrode
        """

        # Index of output electrode
        output_electrode_idx    = self.N_electrodes-1
            
        # Increase the number of nanoparticles by two
        prev_particle_count =   self.N_particles
        self.N_particles    +=  2

        # Find the nanoparticle that is connected to the floating electrode
        adj_np  = np.where(self.net_topology[:,0]==(output_electrode_idx+1))[0][0]

        # Create a new row for the first nanoparticle and set update connection
        new_nn_1    = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
        new_nn_1[1] = adj_np                                                    # Connect to the adjacent nanoparticle
        new_nn_1[2] = self.N_particles-1                                        # Connect to second "new" nanoparticle

        # Create a new row for the first nanoparticle and set the connections
        new_nn_2    = np.full(self.net_topology.shape[1], self.NO_CONNECTION)   # Initialize with placeholders
        new_nn_2[0] = output_electrode_idx+1                                    # Connect to the electrode   
        new_nn_2[1] = self.N_particles-2                                        # Connect to first "new" nanoparticle

        # Add the new nanoparticles and their connections to the network topology
        self.net_topology   = np.vstack((self.net_topology,new_nn_1))
        self.net_topology   = np.vstack((self.net_topology,new_nn_2))

        # Update the adjacent nanoparticle's connection
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,first_free_spot]   = self.net_topology.shape[0]-2
        first_free_spot                             = np.min(np.where(self.net_topology[adj_np,:]==self.NO_CONNECTION))
        self.net_topology[adj_np,0]                 = self.NO_CONNECTION

        # Remove old electrode connection
        self.G.remove_edge(adj_np,-output_electrode_idx-1)
        self.G.remove_edge(-output_electrode_idx-1,adj_np)

        # Add new nanoparticles
        self.G.add_node(prev_particle_count)
        self.G.add_node(prev_particle_count+1)

        # Add new edges
        self.G.add_edge(prev_particle_count,adj_np)
        self.G.add_edge(adj_np,prev_particle_count)
        self.G.add_edge(prev_particle_count,prev_particle_count+1)
        self.G.add_edge(prev_particle_count+1,prev_particle_count)
        self.G.add_edge(prev_particle_count+1,-output_electrode_idx-1)
        self.G.add_edge(-output_electrode_idx-1,prev_particle_count+1)

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

    def return_graph_object(self):
        return self.G
    
    def return_np_pos(self):
        return self.pos

    def validate_network(self) -> bool:
        """
        Validate the network topology.
        
        Checks:
        1. Network connectivity (weak connectivity for directed graphs)
        2. Consistency between net_topology matrix and NetworkX graph
        3. Electrode connections
        
        Returns
        -------
        bool
            True if the network is valid, False otherwise.
            
        Raises
        ------
        ValueError
            If inconsistencies are found in the network topology
        """
        # Check network connectivity (using weak connectivity for directed graphs)
        if not nx.is_weakly_connected(self.G):
            raise ValueError("Network is not fully connected")
            
        # Check consistency between net_topology and graph
        for node in range(self.N_particles):
            graph_neighbors = set(n for n in self.G.neighbors(node) if n >= 0)
            topo_neighbors = set(n for n in self.net_topology[node, 1:] if n != self.NO_CONNECTION)
            if graph_neighbors != topo_neighbors:
                raise ValueError(f"Inconsistency found in connections for node {node}")
                
        # Check electrode connections
        for node in range(self.N_particles):
            electrode = self.net_topology[node, 0]
            if electrode != self.NO_CONNECTION:
                if not self.G.has_edge(node, -(electrode)) or not self.G.has_edge(-(electrode), node):
                    raise ValueError(f"Missing electrode connection for node {node}")
                    
        return True
    
    def export_network(self, filepath: str) -> None:
        """
        Export the network configuration to a file.
        
        This method saves:
        1. Network topology matrix
        2. Node positions
        3. Electrode configurations
        4. Network parameters (N_particles, N_junctions, etc.)
        
        Parameters
        ----------
        filepath : str
            Path to save the network configuration file
        """
        network_data = {
            'net_topology': self.net_topology.tolist(),
            'positions': {str(k): list(v) for k, v in self.pos.items()},
            'N_particles': self.N_particles,
            'N_electrodes': self.N_electrodes,
            'N_junctions': self.N_junctions
        }
        
        if hasattr(self, 'N_x'):
            network_data['N_x'] = self.N_x
            network_data['N_y'] = self.N_y
            network_data['N_z'] = self.N_z
            
        np.save(filepath, network_data, allow_pickle=True)

    def import_network(self, filepath: str) -> None:
        """
        Import a network configuration from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the network configuration file
            
        Raises
        ------
        ValueError
            If the file format is invalid or missing required data
        """
        try:
            network_data = np.load(filepath, allow_pickle=True).item()
            
            # Restore basic attributes
            self.net_topology = np.array(network_data['net_topology'])
            self.pos = {int(k) if k.isdigit() else int(k[1:]) if k.startswith('-') else k: 
                       tuple(v) for k, v in network_data['positions'].items()}
            self.N_particles = network_data['N_particles']
            self.N_electrodes = network_data['N_electrodes']
            self.N_junctions = network_data['N_junctions']
            
            if 'N_x' in network_data:
                self.N_x = network_data['N_x']
                self.N_y = network_data['N_y']
                self.N_z = network_data['N_z']
                
            # Reconstruct the graph
            self.G = nx.DiGraph()
            self.G.add_nodes_from(range(self.N_particles))
            self.G.add_nodes_from(range(-self.N_electrodes, 0))
            
            # Add edges from topology matrix
            for node in range(self.N_particles):
                # Add electrode connections
                if self.net_topology[node, 0] != self.NO_CONNECTION:
                    electrode = -self.net_topology[node, 0]
                    self.G.add_edge(node, electrode)
                    self.G.add_edge(electrode, node)
                    
                # Add nanoparticle connections
                for neighbor in self.net_topology[node, 1:]:
                    if neighbor != self.NO_CONNECTION:
                        self.G.add_edge(node, neighbor)
                        self.G.add_edge(neighbor, node)
                        
            # Validate the imported network
            self.validate_network()
            
        except Exception as e:
            raise ValueError(f"Failed to import network configuration: {str(e)}")
    
    def __str__(self):
        return f"Topology Class with {self.N_particles} particles, {self.N_junctions} junctions.\nNetwork Topology:\n{self.net_topology}"
    
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Cubic Network Topology
    N_x, N_y, N_z   = 3,3,1
    electrode_pos   = [[0,0,0],[1,2,0]]
    cubic_topology  = topology_class()
    cubic_topology.cubic_network(N_x, N_y, N_z)
    cubic_topology.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    print(cubic_topology)
    cubic_topology.add_np_to_output()
    print(cubic_topology)

    
    # Disordered Network Topology
    N_p, N_j        = 20,0
    electrode_pos   = [[-1,-1],[-1,1],[1,1]]
    random_topology = topology_class()
    random_topology.random_network(N_p, N_j)
    random_topology.add_electrodes_to_random_net(electrode_pos)
    random_topology.graph_to_net_topology()
    print(random_topology)
