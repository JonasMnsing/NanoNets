import numpy as np
import topology
from typing import List
from scipy.optimize import least_squares

class electrostatic_class(topology.topology_class):
    """
    Class to setup electrostatic properties of the nanoparticle network.
    This class depends on the topology_class and handles all electrostatic calculations
    including capacitance matrix calculations and charge distributions.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of junctions per nanoparticle
    rng : Generator
        Random number generator
    net_topology : ndarray
        Network topology matrix. Shape (N_particles, N_junctions + 1).
        - First column: connected electrodes (-100 if none)
        - Other columns: connected nanoparticle indices
    eps_r : float
        Relative permittivity of insulating material between nanoparticles (dimensionless)
    eps_s : float
        Relative permittivity of insulating environment/oxide layer (dimensionless)
    radius_vals : ndarray
        Array of nanoparticle radii [nm]
    capacitance_matrix : ndarray
        Network capacitance matrix [aF = 10^-18 F]
    inv_capacitance_matrix : ndarray
        Inverse of capacitance matrix [1/aF]
    charge_vector : ndarray
        Charge values for each nanoparticle [e]
    floating_indices : ndarray
        Indices of floating electrodes in the network

    Physical Constants
    ----------------
    EPSILON_0 : float
        Vacuum permittivity [F/m]
    PI : float
        Mathematical constant π
    ELECTRODE_RADIUS : float
        Default radius used for electrode calculations [nm]
    """
    # Physical constants
    EPSILON_0           = 8.85418781762039e-3  # Vacuum permittivity in aF/nm
    PI                  = 3.14159265359
    ELECTRODE_RADIUS    = 10.0  # nm
    MIN_NP_NP_DISTANCE  = 1.0   # nm
    
    def __init__(self, electrode_type: List[str] = None, seed: int = None)->None:
        """
        Initialize electrostatic properties.

        Parameters
        ----------
        electrode_type : List[str]
            List specifying electrode types ('constant' or 'floating')
        seed : int, optional
            Random seed for reproducibility

        Raises
        ------
        ValueError
            If electrode_type contains invalid values
        """
        super().__init__(seed)
        if not all(t in ['constant', 'floating'] for t in electrode_type):
            raise ValueError("electrode_type must contain only 'constant' or 'floating'")
        self.floating_indices   = np.where(np.array(electrode_type) == 'floating')[0]
        if electrode_type is not None:
            self.electrode_type = np.array(electrode_type)

    def mutal_capacitance_adjacent_spheres_sinh(self, eps_r: float, np_radius1: float, np_radius2: float, N_sum: int = 20) -> float:
        """
        Calculate capacitance between spherical conductors - insulator - spherical conductors setup.

        Parameters
        ----------
        eps_r : float
            Permittivity of insulating material between spheres
        np_radius1 : float
            Radius of first sphere (nanoparticle) [nm]
        np_radius2 : float
            Radius of second sphere (nanoparticle) [nm]
                    
        Returns
        -------
        cap : float
            Capacitance value [aF]
        
        Raises
        ------
        ValueError
            If any input parameters are invalid
        """

        if eps_r <= 0:
            raise ValueError(f"eps_r must be positive, got {eps_r}")
        if np_radius1 <= 0 or np_radius2 <= 0:
            raise ValueError(f"Radii must be positive, got {np_radius1} and {np_radius2}")
                
        # Base factor
        d       = np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE
        factor  = 4 * self.PI * self.EPSILON_0 * eps_r * (np_radius1 * np_radius2) / d
        U_val   = np.arccosh((d**2 - np_radius1**2 - np_radius2**2) / (2*np_radius1*np_radius2))
        cap     = factor * np.sinh(U_val) * np.sum([1/np.sinh(n*U_val) for n in range(1,N_sum+1)])

        return cap

    def mutal_capacitance_adjacent_spheres(self, eps_r: float, np_radius1: float, np_radius2: float) -> float:
        """
        Calculate capacitance between spherical conductors - insulator - spherical conductors setup.
        Uses a third order Taylor Expansion based on the Image Charge method.

        Parameters
        ----------
        eps_r : float
            Permittivity of insulating material between spheres
        np_radius1 : float
            Radius of first sphere (nanoparticle) [nm]
        np_radius2 : float
            Radius of second sphere (nanoparticle) [nm]
                    
        Returns
        -------
        cap : float
            Capacitance value [aF]
        
        Raises
        ------
        ValueError
            If any input parameters are invalid
        """
        if eps_r <= 0:
            raise ValueError(f"eps_r must be positive, got {eps_r}")
        if np_radius1 <= 0 or np_radius2 <= 0:
            raise ValueError(f"Radii must be positive, got {np_radius1} and {np_radius2}")
                
        # Base factor
        d       = np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE
        factor  = 4 * self.PI * self.EPSILON_0 * eps_r * (np_radius1 * np_radius2) / d
        
        # Terms of the Taylor expansion
        term1       = 1.0
        term2       = (np_radius1 * np_radius2) / (d**2 - 2*np_radius1*np_radius2)
        denominator = d**4 - 4*(d**2)*np_radius1*np_radius2 + 3*(np_radius1**2)*(np_radius2**2)
        term3       = ((np_radius1**2)*(np_radius2**2)) / denominator
        cap         = factor * (term1 + term2 + term3)
                    
        return cap

    def self_capacitance_sphere(self, eps_s: float, np_radius: float) -> float:
        """
        Calculate self capacitance of a sphere in an insulating environment.

        Uses the formula C = 4πε₀εᵣr where:
        - eps_0 is the vacuum permittivity
        - eps_s is the relative permittivity of the environment
        - r is the radius of the sphere

        Parameters
        ----------
        eps_s : float
            Relative permittivity of insulating environment (dimensionless)
        np_radius : float
            Radius of sphere (nanoparticle) [nm]
        
        Returns
        -------
        cap : float
            Capacitance value [aF]

        Raises
        ------
        ValueError
            If eps_s <= 0 or np_radius <= 0
        """
        if eps_s <= 0:
            raise ValueError(f"Environment permittivity must be positive, got {eps_s}")
        if np_radius <= 0:
            raise ValueError(f"Radius must be positive, got {np_radius}")

        factor = 4 * self.PI * self.EPSILON_0 * eps_s
        cap = factor * np_radius

        return cap
        
    def init_nanoparticle_radius(self, mean_radius=10.0, std_radius=0.0)->None:
        """
        Sample radii for all nanoparticles from |Gaussian(mean_radius, std_radius)|

        Parameters
        ----------
        mean_radius : float
            Average nanoparticle radius [nm]
        std_radius : float
            Radius standard deviation [nm]
            
        Raises
        ------
        ValueError
            If mean_radius <= 0 or std_radius < 0
        """
        if mean_radius <= 0:
            raise ValueError(f"Mean radius must be positive, got {mean_radius}")
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}")

        self.radius_vals = np.abs(self.rng.normal(loc=mean_radius, scale=std_radius, size=self.N_particles))

    def update_nanoparticle_radius(self, nanoparticles: list, mean_radius: float = 10.0, std_radius: float = 0.0) -> None:
        """
        Update radii of specific nanoparticles in the network.

        Parameters
        ----------
        nanoparticles : list
            List of nanoparticle indices to update
        mean_radius : float, optional
            New mean radius for selected nanoparticles [nm], by default 10.0
        std_radius : float, optional
            Standard deviation of radius distribution [nm], by default 0.0

        Raises
        ------
        ValueError
            If mean_radius <= 0 or std_radius < 0
            If any nanoparticle index is invalid
        RuntimeError
            If radius_vals hasn't been initialized
        """
        if not hasattr(self, 'radius_vals'):
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")
            
        if mean_radius <= 0:
            raise ValueError(f"Mean radius must be positive, got {mean_radius}")
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}")
            
        invalid_indices = [i for i in nanoparticles if i < 0 or i >= self.N_particles]
        if invalid_indices:
            raise ValueError(f"Invalid nanoparticle indices: {invalid_indices}")
            
        # Update radii for specified particles
        self.radius_vals[nanoparticles] = np.abs(
            self.rng.normal(loc=mean_radius, scale=std_radius, size=len(nanoparticles))
        )
        
    @staticmethod                    
    def _seg_seg_dist(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:

        u       = p2 - p1
        v       = q2 - q1
        w       = p1 - q1
        a, b, c = u.dot(u), u.dot(v), v.dot(v)
        d, e    = u.dot(w), v.dot(w)
        D       = a*c - b*b

        if D < 1e-8:
            s = 0.0
            t = e / c if c > 1e-8 else 0.0
        else:
            s = ( b*e - c*d) / D
            t = ( a*e - b*d) / D

        s   = np.clip(s, 0.0, 1.0)
        t   = np.clip(t, 0.0, 1.0)
        cp1 = p1 + s*u
        cp2 = q1 + t*v

        return np.linalg.norm(cp1 - cp2)
    
    # def position_planar(self, w_neigh: float = 1.0, w_minsep: float = 10.0,
    #                             w_planar: float = 1e5, margin: float = 0.0) -> None:
    #     """
    #     Compute a planar, non-overlapping layout for the current lattice graph.

    #     Parameters
    #     ----------
    #     w_neigh : float
    #         Spring weight pulling neighbor pairs to minimal gap.
    #     w_minsep : float
    #         Barrier weight enforcing minimal gap for all pairs.
    #     w_planar : float
    #         Penalty weight forbidding edge crossings.
    #     margin : float
    #         Minimum clearance between non-adjacent edges (0 forbids crossing).
    #     """

    #     # Save existing electrode positions
    #     electrode_positions = {n: self.pos[n] for n in self.pos if n < 0}
        
    #     # Build adjacency list from net_topology (neighbors only):
    #     adj_array = self.net_topology[:, 1:]

    #     # Call pack-down algorithm
    #     coords = self._pack_down_pull_min_all(adj_array, w_neigh, w_minsep, w_planar, margin)
        
    #     # Update nanoparticle positions
    #     self.pos = {i: tuple(coords[i]) for i in range(self.N_particles)}
    #     # Restore electrode positions
    #     for n, p in electrode_positions.items():
    #         self.pos[n] = p

    #     # Compute full distance matrix
    #     diff                = coords[:,None,:]-coords[None,:,:]
    #     dist_matrix         = np.linalg.norm(diff,axis=2)
    #     self.dist_matrix    = dist_matrix

    #     # Build electrode-to-particle distance matrix
    #     electrodes      = sorted(electrode_positions.keys(), key=lambda x: -x)
    #     electrode_dist  = np.zeros((self.N_electrodes, self.N_particles))
    #     for idx, enode in enumerate(electrodes):
    #         ex, ey = electrode_positions[enode]
    #         for j in range(self.N_particles):
    #             px, py                  = coords[j]
    #             electrode_dist[idx, j]  = np.hypot(ex - px, ey - py)
    #     self.electrode_dist_matrix = electrode_dist

    # def _pack_down_pull_min_all(self, adj_array: np.ndarray, w_neigh: float, w_minsep: float, w_planar: float, margin: float) -> np.ndarray:
    #     """
    #     Internal helper: grid-init then pull-down with planar, non-overlap constraints.
    #     """

    #     # Undirected neighbor edges
    #     neigh_lists = [np.unique(adj_array[i][adj_array[i] >= 0].astype(int)) for i in range(self.N_particles)]
    #     edges       = [(i, j) for i in range(self.N_particles) for j in neigh_lists[i] if j > i]

    #     # Grid spacing D
    #     min_neigh   = np.array([self.radius_vals[i] + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE for i, j in edges])
    #     D           = min_neigh.max()

    #     # Initial grid coords (requires N perfect square)
    #     L       = int(np.round(np.sqrt(self.N_particles)))
    #     grid_xy = np.column_stack((np.arange(self.N_particles) % L, np.arange(self.N_particles) // L))
    #     x0      = (grid_xy * D).ravel()

    #     # All pairs for global min separation
    #     all_pairs = [(i, j) for i in range(self.N_particles) for j in range(i+1, self.N_particles)]

    #     # Edge-pairs for planarity
    #     edge_pairs = []
    #     for a, b in edges:
    #         for c, d in edges:
    #             if {a, b}.isdisjoint({c, d}) and a < b and c < d:
    #                 if (c, d, a, b) not in edge_pairs:
    #                     edge_pairs.append((a, b, c, d))

    #     # Precompute min distances
    #     min_all = { (i, j): self.radius_vals[i] + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE for (i, j) in all_pairs }

    #     def residuals(x):
    #         X   = x.reshape(self.N_particles, 2)
    #         res = []

    #         # spring on neighbors
    #         for (i, j) in edges:
    #             d = np.linalg.norm(X[i] - X[j])
    #             res.append(w_neigh * (d - min_all[(i, j)]))

    #         # global min separation
    #         for (i, j) in all_pairs:
    #             d   = np.linalg.norm(X[i] - X[j])
    #             pen = max(0.0, min_all[(i, j)] - d)
    #             res.append(w_minsep * pen)

    #         # planarity hinge
    #         for (i, j, k, l) in edge_pairs:
    #             dseg    = self._seg_seg_dist(X[i], X[j], X[k], X[l])
    #             pen     = max(0.0, margin - dseg)
    #             res.append(w_planar * pen)

    #         return np.array(res)

    #     sol = least_squares(residuals, x0, method='trf')
    #     return sol.x.reshape(self.N_particles, 2)
    
    def position_planar(self, w_neigh: float = 1.0, w_minsep: float = 10.0,
                        w_planar: float = 1e5, margin: float = 1e-3) -> np.ndarray:
        """
        Optimize planar layout for nanoparticles, then reposition electrodes at fixed gap.

        Returns
        -------
        dist_matrix : ndarray
            N_particles x N_particles matrix of NP-NP distances
        """
        # 0) Backup old positions for electrode offsets
        self.pos_old = self.pos.copy()
        # Identify electrode nodes
        electrode_nodes = [n for n in self.pos_old if isinstance(n, int) and n < 0]
        # 1) Solve nanoparticles only
        radii = self.radius_vals.copy()
        nanop_adj = self.net_topology[:,1:]
        Np = self.N_particles
        # Build adjacency mask for nanoparticles
        neighbors = []
        for i in range(Np):
            neigh = nanop_adj[i][nanop_adj[i]>=0].astype(int)
            neighbors.append(np.unique(neigh))
        max_deg = max((len(ne) for ne in neighbors), default=0)
        adj_mask = -np.ones((Np, max_deg), dtype=int)
        for i, ne in enumerate(neighbors): adj_mask[i,:len(ne)] = ne
        # Initial guess from old nanoparticle positions
        x0 = np.zeros((Np,2))
        for i in range(Np): x0[i] = self.pos_old.get(i, (0.0,0.0))
        # Perform spring/hinge solver
        coords = self._pack_down_pull_all(adj_mask,
                                          radii,
                                          w_neigh,
                                          w_minsep,
                                          w_planar,
                                          margin,
                                          x0.ravel())
        # Update nanoparticle positions
        for i in range(Np): self.pos[i] = tuple(coords[i])
        # 2) Reposition electrodes based on initial offsets
        for e_node in electrode_nodes:
            old_e_pos = np.array(self.pos_old[e_node])
            # find connected nanoparticle index
            p_indices = np.where(self.net_topology[:,0] == -e_node)[0]
            if len(p_indices) != 1:
                continue
            p = p_indices[0]
            # new particle position
            p_pos = coords[p]
            # old particle position
            old_p_pos = np.array(self.pos_old[p])
            # direction from p to electrode
            dir_vec = old_e_pos - old_p_pos
            if np.allclose(dir_vec, 0):
                dir_vec = np.array([1.0, 0.0])
            u = dir_vec / np.linalg.norm(dir_vec)
            # compute new electrode position at exact gap
            gap = radii[p] + self.ELECTRODE_RADIUS + 1
            new_epos = p_pos + u * gap
            self.pos[e_node] = tuple(new_epos)
        # 3) Nanoparticle distance matrix
        diff = coords[:, None, :] - coords[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        self.dist_matrix = dist_matrix
        # 4) Electrode-particle distances
        Ne = len(electrode_nodes)
        el_dist = np.zeros((Ne, Np))
        for idx, e_node in enumerate(electrode_nodes):
            epos = np.array(self.pos[e_node])
            d = coords - epos
            el_dist[idx] = np.linalg.norm(d, axis=1)
        self.electrode_dist_matrix = el_dist
        return dist_matrix

    def _pack_down_pull_all(self,
                            adj_array: np.ndarray,
                            radii: np.ndarray,
                            w_neigh: float,
                            w_minsep: float,
                            w_planar: float,
                            margin: float,
                            x0: np.ndarray) -> np.ndarray:
        """Pack-down solver for all nodes, using x0 initial guess."""
        N = len(radii)
        adj = np.asarray(adj_array)
        neighbors = [adj[i][adj[i]>=0].astype(int) for i in range(N)]
        edges = [(i,j) for i in range(N) for j in neighbors[i] if j>i]
        # determine minimal spring length
        if edges:
            min_edge = np.array([radii[i]+radii[j]+1 for i,j in edges])
            D = min_edge.max()
        else:
            D = 1.0
        # prepare pairs
        all_pairs = [(i,j) for i in range(N) for j in range(i+1,N)]
        edge_pairs = [(a,b,c,d)
                      for (a,b) in edges for (c,d) in edges
                      if {a,b}.isdisjoint({c,d}) and a<b and c<d]
        min_all = {(i,j): radii[i]+radii[j]+1 for (i,j) in all_pairs}
        def residuals(x):
            X = x.reshape(N,2)
            res = []
            for (i,j) in edges:
                d = np.linalg.norm(X[i]-X[j])
                res.append(w_neigh*(d - min_all[(i,j)]))
            for (i,j) in all_pairs:
                d = np.linalg.norm(X[i]-X[j]); pen = max(0.0, min_all[(i,j)]-d)
                res.append(w_minsep*pen)
            for (i,j,k,l) in edge_pairs:
                dseg = self._seg_seg_dist(X[i],X[j],X[k],X[l]); pen = max(0.0, margin-dseg)
                res.append(w_planar*pen)
            return np.array(res)
        sol = least_squares(residuals, x0, method='trf')
        return sol.x.reshape(N,2)


    
    def calc_capacitance_matrix(self, eps_r: float = 2.6, eps_s: float = 3.9)->None:
        """
        Calculate the capacitance matrix of the nanoparticle network.

        Parameters
        ----------
        eps_r : float
            Relative permittivity of insulating material between nanoparticles
        eps_s : float
            Relative permittivity of insulating environment (oxide layer)
            
        Raises
        ------
        ValueError
            If physical parameters are invalid
        RuntimeError
            If matrix inversion fails
        """
        if not hasattr(self, 'radius_vals'):
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")

        # Initialize capacitance matrix
        self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
        self.eps_r              = eps_r
        self.eps_s              = eps_s

        # Loop over each particle to calculate capacitance contributions
        for i in range(self.N_particles):
            C_sum = 0.0
            # Iterate over the junctions of the current nanoparticle
            for j in range(self.N_junctions+1):
                neighbor = self.net_topology[i,j]

                # Skip if the neighbor is invalid
                if neighbor == self.NO_CONNECTION:
                    continue

                # Capacitance with the electrode (j == 0)
                if (j == 0):
                    if neighbor-1 not in self.floating_indices:
                        # Add electrode capacitance (using mutual capacitance formula)
                        C_sum += self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)

                else:
                    # Calc mutual capacitance
                    val = self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[int(neighbor)])
                    self.capacitance_matrix[i,int(neighbor)] = -val
                    C_sum += val

            # Self Capacitance
            C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            
            # Set diagonal element (total capacitance)
            self.capacitance_matrix[i,i] = C_sum
        
        # Calculate the inverse capacitance matrix
        try:
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e

    def init_charge_vector(self, voltage_values: np.array)->None:
        """
        Initialize the charge vector based on electrode voltages.

        Parameters
        ----------
        voltage_values : array
           Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
           where V_G is the gate voltage

        Raises
        ------
        ValueError
            If voltage array length doesn't match number of electrodes + 1
            If floating electrode voltages are non-zero
        RuntimeError
            If capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

        if len(voltage_values) != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {len(voltage_values)}")

        floating_voltages = [voltage_values[i] for i in self.floating_indices]
        if any(v != 0 for v in floating_voltages):
            raise ValueError("Floating electrode voltages must be initialized to zero")

        # Initialize charge vector
        self.charge_vector = np.zeros(self.N_particles)

        # Iterate over all nanoparticles
        for i in range(self.N_particles):
            electrode_index = int(self.net_topology[i,0] - 1)

            if self.net_topology[i,0] != self.NO_CONNECTION:
                # If connected to an electrode, calculate charge from electrode voltage
                if electrode_index not in self.floating_indices:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
                else:
                    C_lead  = 0.0
                
                C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                self.charge_vector[i] = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
            else:
                # If not connected to an electrode
                C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                self.charge_vector[i] = voltage_values[-1] * C_self

    def get_charge_vector_offset(self, voltage_values : np.array)->np.array:
        """
        Calculate the charge vector offset induced by electrode voltages.
        
        This method computes the charges induced on nanoparticles by the electrodes
        without reinitializing the entire charge vector. This is useful when you want
        to update only the portion of charges that changes due to external voltage changes.

        Parameters
        ----------
        voltage_values : np.array
            Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
            where V_G is the gate voltage

        Returns
        -------
        np.array
            Charge offset vector [aC] representing charges induced by electrodes
        """

        offset = np.zeros(self.N_particles)

        # For each charge
        for i in range(self.N_particles):
            electrode_index = int(self.net_topology[i,0] - 1)

            if self.net_topology[i,0] != self.NO_CONNECTION:
                # If connected to an electrode, calculate charge from electrode voltage
                if electrode_index not in self.floating_indices:
                    C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
                else:
                    C_lead  = 0.0
                
                C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                offset[i]   = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
            else:
                # If not connected to an electrode
                C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
                offset[i]   = voltage_values[-1] * C_self

        return offset

    def delete_n_junctions(self, n: int) -> None:
        """Delete n random junctions in the network.
        
        Only deletes junctions between nanoparticles that are not connected 
        to electrodes to preserve network functionality.
        
        Parameters
        ----------
        n : int
            Number of junctions to delete
            
        Raises
        ------
        ValueError
            If n is negative or larger than available junctions
        RuntimeError
            If no valid junctions can be found to delete
        """
        if n < 0:
            raise ValueError(f"Number of junctions to delete must be non-negative, got {n}")
            
        # Count available junctions (between non-electrode connected particles)
        available_junctions = 0
        for i in range(self.N_particles):
            if self.net_topology[i,0] == self.NO_CONNECTION:  # Not connected to electrode
                available_junctions += sum(1 for j in self.net_topology[i,1:] if j != self.NO_CONNECTION)
        available_junctions //= 2  # Each junction is counted twice
        
        if n > available_junctions:
            raise ValueError(f"Cannot delete {n} junctions, only {available_junctions} available")

        for i in range(n):
            max_attempts = 5000
            attempt = 0
            
            while attempt < max_attempts:
                np1 = self.rng.integers(0, self.N_particles)
                np2 = self.rng.integers(1, self.N_junctions + 1)
                
                # Check if this is a valid junction to delete
                if (self.net_topology[np1,np2] != self.NO_CONNECTION and 
                    self.net_topology[np1,0] == self.NO_CONNECTION):
                    # Found a valid junction, delete it
                    np1_2 = int(self.net_topology[np1,np2])
                    np2_2 = np.where(self.net_topology[np1_2,1:] == np1)[0][0]
                    
                    # Remove junction from topology matrix
                    self.net_topology[np1,np2] = self.NO_CONNECTION
                    self.net_topology[np1_2,np2_2+1] = self.NO_CONNECTION
                    
                    # Remove edges from graph
                    self.G.remove_edge(np1, np1_2)
                    self.G.remove_edge(np1_2, np1)
                    break
                    
                attempt += 1
                
            if attempt >= max_attempts:
                raise RuntimeError(f"Could not find valid junction to delete after {max_attempts} attempts")
    
    def return_charge_vector(self) -> np.array:
        """Get the current charge vector of the network.

        Returns
        -------
        np.array
            Charge values for each nanoparticle [e]
            
        Raises
        ------
        RuntimeError
            If charge vector hasn't been initialized
        """
        if not hasattr(self, 'charge_vector'):
            raise RuntimeError("Charge vector not initialized. Call init_charge_vector first.")
        return self.charge_vector
    
    def return_capacitance_matrix(self) -> np.array:
        """Get the network capacitance matrix.

        Returns
        -------
        np.array
            Array containing network capacitance values [aF]
            Shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.capacitance_matrix
        
    def return_inv_capacitance_matrix(self) -> np.array:
        """Get the inverse capacitance matrix.

        Returns
        -------
        np.array
            Inverse of capacitance matrix [1/aF]
            Shape: (N_particles, N_particles)
            
        Raises
        ------
        RuntimeError
            If inverse capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'inv_capacitance_matrix'):
            raise RuntimeError("Inverse capacitance matrix not calculated. Call calc_capacitance_matrix first.")
        return self.inv_capacitance_matrix

    def return_net_topology(self) -> np.array:
        """Get the network topology matrix.

        Returns
        -------
        np.array
            Network topology matrix where:
            - Rows represent nanoparticles
            - First column stores connected electrodes
            - Second to last columns store connected nanoparticles
            - NO_CONNECTION (-100) indicates no connection
            
        See Also
        --------
        topology_class.return_net_topology : Parent class method
        """
        return super().return_net_topology()

    def return_dist_matrix(self) -> np.ndarray:
        return self.dist_matrix
    
    def return_electrode_dist_matrix(self) -> np.ndarray:
        return self.electrode_dist_matrix
            
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Parameter
    N_x, N_y, N_z       = 3,3,1
    electrode_pos       = [[0,0,0],[1,2,0]]
    radius, radius_std  = 10.0, 0.0
    eps_r, eps_s        = 2.6, 3.9
    voltage_values      = [0.8,0.0,0.0]
    electrode_type      = ['constant','floating']
    high_cap_nps        = [N_x*N_y]
    high_cap            = 1e2

    # Electrostatic
    cubic_electrostatic = electrostatic_class(electrode_type)
    cubic_electrostatic.cubic_network(N_x, N_y, N_z)
    cubic_electrostatic.set_electrodes_based_on_pos(electrode_pos, N_x, N_y)
    cubic_electrostatic.add_np_to_output()
    cubic_electrostatic.init_nanoparticle_radius(radius, radius_std)
    cubic_electrostatic.update_nanoparticle_radius(high_cap_nps, high_cap)
    cubic_electrostatic.calc_capacitance_matrix(eps_r, eps_s)
    cubic_electrostatic.init_charge_vector(voltage_values)

    capacitance_matrix      = cubic_electrostatic.return_capacitance_matrix()
    inv_capacitance_matrix  = cubic_electrostatic.return_inv_capacitance_matrix()
    charge_vector           = cubic_electrostatic.return_charge_vector()

    print(cubic_electrostatic)
    print("Capacitance Matrix:\n", np.round(capacitance_matrix,2))
    print("Initial Charge Vector:\n", np.round(charge_vector,2))
    print("Graph Positions:\n", cubic_electrostatic.pos)
    print("Graph Nodes:\n", cubic_electrostatic.G.nodes)
    print("Graph Edges:\n", cubic_electrostatic.G.edges)
