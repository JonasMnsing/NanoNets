import numpy as np
import networkx as nx
from . import topology
from typing import List, Optional
from shapely.geometry import Point, LineString
from shapely.ops import unary_union, nearest_points
from itertools import combinations

class NanoparticleElectrostatic(topology.NanoparticleTopology):
    """
    Extends NanoparticleTopology with electrostatic modeling and physical parameters
    for 2D nanoparticle networks.

    This class enables computation of capacitance matrices, induced charges, 
    polydisperse packing, and physically accurate (planar) network geometries.
    Supports both regular lattice and random (Delaunay) planar networks,
    with advanced handling of boundary conditions for 'constant' (voltage-biased)
    and 'floating' electrodes.

    Attributes
    ----------
    Inherited from NanoparticleTopology:
        N_particles : int
            Number of nanoparticles.
        N_electrodes : int
            Number of electrodes.
        N_junctions : int
            Number of junctions per nanoparticle (target/average).
        G : nx.DiGraph
            NetworkX directed graph object.
        pos : dict
            Mapping of node indices to 2D positions.
        net_topology : np.ndarray
            Matrix encoding connections and electrode links.

    Added by this class:
        electrode_type : np.ndarray of str
            Type for each electrode ('constant' or 'floating').
        floating_indices : np.ndarray of int
            Indices of floating electrodes.
        radius_vals : np.ndarray of float
            Radii of nanoparticles [nm].
        capacitance_matrix : np.ndarray
            Full network capacitance matrix [aF].
        inv_capacitance_matrix : np.ndarray
            Inverse capacitance matrix [1/aF].
        charge_vector : np.ndarray
            Vector of induced NP charges [aC].
        self_capacitance : np.ndarray
            Self-capacitance per NP [aF].
        electrode_capacitance_matrix : np.ndarray
            Capacitance matrix between electrodes and NPs [aF].
        dist_matrix : np.ndarray
            Pairwise NP-NP distance matrix [nm].
        electrode_dist_matrix : np.ndarray
            Pairwise NP-electrode distance matrix [nm].

    Physical Constants
    ------------------
        EPSILON_0 : float
            Vacuum permittivity [aF/nm]
        PI : float
            Pi
        ELECTRODE_RADIUS : float
            Electrode radius [nm]
        MIN_NP_NP_DISTANCE : float
            Minimum center-to-center distance between NPs [nm]
        MIN_NP_RADIUS : float
            Minimum allowed NP radius [nm]

    Methods
    -------
        init_nanoparticle_radius(mean, std)
            Initialize all NP radii (with optional polydispersity).
        update_nanoparticle_radius(indices, mean, std)
            Update radii for selected NPs.
        pack_planar_circles(...)
            Adjust NP positions to remove overlaps, keeping network planar.
        calc_capacitance_matrix(eps_r, eps_s)
            Calculate the full NP network capacitance matrix.
        calc_electrode_capacitance_matrix()
            Compute NP-electrode and self-capacitance terms.
        init_charge_vector(voltage_values)
            Set induced NP charges based on electrode voltages.
        get_charge_vector()
            Return induced NP charge vector [aC].
        get_capacitance_matrix()
            Return full NP capacitance matrix [aF].
        get_self_capacitance()
            Return self-capacitance per NP [aF].
        get_dist_matrix()
            Return NP-NP distance matrix [nm].
        delete_n_junctions(n)
            Randomly remove n junctions (ensuring network remains connected).

    Notes
    -----
    - All distances/radii in nanometers (nm); all capacitances in attofarads (aF).
    - The class enforces physical plausibility (no overlaps, realistic capacitances).
    """
    # Physical constants
    EPSILON_0           = 8.85418781762039e-3  # aF/nm, vacuum permittivity
    PI                  = 3.14159265359
    ELECTRODE_RADIUS    = 10.0  # nm
    MIN_NP_NP_DISTANCE  = 1.0   # nm
    MIN_NP_RADIUS       = 5.0   # nm, minimum allowed nanoparticle radius
    
    def __init__(self, electrode_type: Optional[List[str]] = None, seed: Optional[int] = None) -> None:
        """
        Initialize the NanoparticleElectrostatic network.

        Parameters
        ----------
        electrode_type : List[str], optional
            List of electrode types, each element should be 'constant' or 'floating'.
            Length must match number of electrodes to be attached to the network.
            If None, no electrode types are set initially.
        seed : int, optional
            Seed for the random number generator (for reproducibility).

        Raises
        ------
        ValueError
             If electrode_type contains values other than 'constant' or 'floating'.
        """
        super().__init__(seed)
        self.electrode_type     = None
        self.floating_indices   = np.array([], dtype=int)
        if electrode_type is not None:
            arr = np.array(electrode_type)
            if not np.all(np.isin(arr, ['constant', 'floating'])):
                raise ValueError("electrode_type must contain only 'constant' or 'floating'")
            self.electrode_type     = arr
            self.floating_indices   = np.where(arr == 'floating')[0]

    def mutual_capacitance_adjacent_spheres(self, eps_r: float, np_radius1: float, np_radius2: float, distance: float, N_sum: int = 50) -> float:
        """
        Compute the mutual capacitance between two adjacent spherical nanoparticles
        separated by an insulator, using an exact series solution.

        The capacitance is computed in attofarads [aF]. All distances are in nanometers [nm].

        Parameters
        ----------
        eps_r : float
            Relative permittivity of the insulating material between spheres (dimensionless)
        np_radius1 : float
            Radius of first sphere [nm]
        np_radius2 : float
            Radius of second sphere [nm]
        distance : float
            Center-to-center distance between spheres [nm]
        N_sum : int, optional
            Number of terms in the series expansion (default: 50)

        Returns
        -------
        float
            Capacitance value [aF]
        
        Raises
        ------
        ValueError
            If any parameter is non-positive.

        Notes
        -----
        The function enforces a minimum center-to-center distance to avoid unphysical overlaps.
        """

        if eps_r <= 0:
            raise ValueError(f"Relative permittivity eps_r must be positive, got {eps_r}.")
        if np_radius1 <= 0 or np_radius2 <= 0:
            raise ValueError(f"Radii must be positive, got {np_radius1} and {np_radius2}.")
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}.")
        if N_sum < 1:
            raise ValueError("N_sum must be at least 1.")
        
        # Minimum separation to avoid overlap
        min_sep = np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE
        if distance < min_sep:
            distance = min_sep

        factor  = 4 * self.PI * self.EPSILON_0 * eps_r * (np_radius1 * np_radius2) / distance
        arg     = (distance**2 - np_radius1**2 - np_radius2**2) / (2 * np_radius1 * np_radius2)
        U_val   = np.arccosh(arg)
        s       = np.sum([1.0 / np.sinh(n * U_val) for n in range(1 ,N_sum + 1)])
        cap     = factor * np.sinh(U_val) * s

        return cap

    def self_capacitance_sphere(self, eps_s: float, np_radius: float) -> float:
        """
        Compute the self-capacitance of a single spherical nanoparticle in an infinite homogeneous dielectric.

        Parameters
        ----------
        eps_s : float
            Relative permittivity of the surrounding medium [dimensionless]
        np_radius : float
            Radius of the nanoparticle [nm]
        
        Returns
        -------
        float
            Capacitance [aF]

        Raises
        ------
        ValueError
            If eps_s <= 0 or np_radius <= 0
        """
        if eps_s <= 0:
            raise ValueError(f"Relative permittivity eps_s must be positive, got {eps_s}.")
        if np_radius <= 0:
            raise ValueError(f"Radius must be positive, got {np_radius}.")

        factor  = 4 * self.PI * self.EPSILON_0 * eps_s
        cap     = factor * np_radius

        return cap
        
    def init_nanoparticle_radius(self, mean_radius: float = 10.0, std_radius: float = 0.0) -> None:
        """
        Initialize radii for all nanoparticles, ensuring all radii are >= MIN_NP_RADIUS.

        Samples from Normal(mean_radius, std_radius), resampling any values < MIN_NP_RADIUS.

        All values are in nanometers [nm].

        Parameters
        ----------
        mean_radius : float, optional
            Mean nanoparticle radius [nm]. Must be > 0. (Default: 10.0)
        std_radius : float, optional
            Standard deviation for radii [nm]. Must be >= 0. (Default: 0.0)

        Raises
        ------
        ValueError
            If mean_radius <= 0 or std_radius < 0.
            If no nanoparticles are present in the network.
        """
        if mean_radius <= 0:
            raise ValueError(f"Mean radius must be positive, got {mean_radius}.")
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}.")
        if getattr(self, 'N_particles', 0) <= 0:
            raise ValueError("No nanoparticles defined. Initialize the network before setting radii.")

        if std_radius == 0:
            self.radius_vals = np.full(self.N_particles, mean_radius)
        else:
            radius = self.rng.normal(loc=mean_radius, scale=std_radius, size=self.N_particles)
            # Rejection sampling for values < min_radius
            while np.any(radius < self.MIN_NP_RADIUS):
                n_bad = np.sum(radius < self.MIN_NP_RADIUS)
                radius[radius < self.MIN_NP_RADIUS] = self.rng.normal(loc=mean_radius, scale=std_radius, size=n_bad)
            self.radius_vals = radius

    def update_nanoparticle_radius(self, nanoparticles: list, mean_radius: float = 10.0, std_radius: float = 0.0) -> None:
        """
        Update the radii of specific nanoparticles in the network.

        The new radii are sampled from a normal distribution (mean_radius, std_radius),
        but any values smaller than MIN_NP_RADIUS are resampled (truncated normal).

        All radii are in nanometers [nm].

        Parameters
        ----------
        nanoparticles : list of int
            Indices of nanoparticles to update.
        mean_radius : float, optional
            Mean radius for the new values [nm] (default: 10.0).
        std_radius : float, optional
            Standard deviation of radius [nm] (default: 0.0).

        Raises
        ------
        ValueError
            If mean_radius <= 0, std_radius < 0, or if any index is out of bounds.
        RuntimeError
            If radius_vals has not been initialized (call init_nanoparticle_radius first).
        """
        if not hasattr(self, 'radius_vals'):
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")

        if mean_radius <= 0:
            raise ValueError(f"Mean radius must be positive, got {mean_radius}.")
        if std_radius < 0:
            raise ValueError(f"Standard deviation must be non-negative, got {std_radius}.")

        invalid_indices = [i for i in nanoparticles if i < 0 or i >= self.N_particles]
        if invalid_indices:
            raise ValueError(f"Invalid nanoparticle indices: {invalid_indices}")
            
        invalid_indices = [i for i in nanoparticles if i < 0 or i >= self.N_particles]
        if invalid_indices:
            raise ValueError(f"Invalid nanoparticle indices: {invalid_indices}")

        if std_radius == 0:
            self.radius_vals[nanoparticles] = mean_radius
        else:
            n_update = len(nanoparticles)
            # Draw initial sample
            new_radii = self.rng.normal(loc=mean_radius, scale=std_radius, size=n_update)
            # Rejection sampling for truncated normal
            while np.any(new_radii < self.MIN_NP_RADIUS):
                n_bad = np.sum(new_radii < self.MIN_NP_RADIUS)
                new_radii[new_radii < self.MIN_NP_RADIUS] = self.rng.normal(loc=mean_radius, scale=std_radius, size=n_bad)
            self.radius_vals[nanoparticles] = new_radii
               
    def pack_planar_circles(self, max_iter: int = 50, alpha: float = 0.05, tol: float = 1e-3, shrink_tol: float = 1e-6) -> None:
        """
        Adjusts `self.pos` so that:
        1. All nanoparticle circles (of varying radii) are non-overlapping.
        2. All graph edges remain straight and non-crossing (planar).
        3. The final packing is as dense as possible (global shrink-to-fit).
        4. Electrodes are placed outside without overlap.

        Uses force-directed relaxation, then binary shrink-to-fit, then assigns electrodes.

        Parameters
        ----------
        max_iter : int
            Maximum number of repulsion/attraction iterations.
        alpha : float
            Strength of edge-based attraction (0 < alpha < 1).
        tol : float
            Overlap tolerance for circles.
        shrink_tol : float
            Precision for the final shrink step.

        Raises
        ------
        RuntimeError
            If required attributes are not initialized.
        """
        # Check necessary attributes
        if not hasattr(self, "pos") or not self.pos:
            raise RuntimeError("Particle positions not initialized.")
        if not hasattr(self, "radius_vals"):
            raise RuntimeError("Particle radii not initialized.")
        if not hasattr(self, "G"):
            raise RuntimeError("Graph object not initialized.")
                
        # Extract the subgraph of nanoparticles only, with their initial positions
        G_np = self.G.copy()
        for e in range(1, self.N_electrodes + 1):
            G_np.remove_node(-e)
        pos_np = {n: np.array(self.pos[n], dtype=float) for n in self.pos if n >= 0}

        # Scale to avoid initial overlaps
        min_scale = 1.0
        for u, v in G_np.edges():
            p_u, p_v    = pos_np[u], pos_np[v]
            dist        = np.linalg.norm(p_u - p_v)
            required    = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
            if dist > 0:
                min_scale = max(min_scale, required / dist)
        if min_scale > 1.0:
            for n in pos_np:
                pos_np[n] *= min_scale

        # Local relaxation (repel overlaps, attract edges, fix crossings)
        for _ in range(max_iter):
            max_overlap = 0.0

            # A) Repel overlapping circles
            for u, v in combinations(G_np.nodes(), 2):
                p_u, p_v    = pos_np[u], pos_np[v]
                req         = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
                delta       = p_u - p_v
                d           = np.hypot(*delta)
                overlap     = req - d
                if overlap > tol:
                    max_overlap = max(max_overlap, overlap)
                    if d < 1e-6:
                        direction = np.random.randn(2)
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = delta / d
                    shift       = 0.5 * overlap * direction
                    pos_np[u]   +=  shift
                    pos_np[v]   -=  shift

            # B) Attract Neighbors
            for u, v in G_np.edges():
                delta   = pos_np[v] - pos_np[u]
                d       = np.hypot(*delta)
                desired = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
                if d > desired + tol:
                    shift_amount    = alpha * (d - desired)
                    direction       = delta / d
                    pos_np[u]       += shift_amount * direction
                    pos_np[v]       -= shift_amount * direction

            # C) Untangle any crossing edges (should be rare)
            for (u1, v1), (u2, v2) in combinations(G_np.edges(), 2):
                if len({u1, v1, u2, v2}) < 4:
                    continue
                seg1 = LineString([pos_np[u1], pos_np[v1]])
                seg2 = LineString([pos_np[u2], pos_np[v2]])
                if seg1.crosses(seg2):
                    avg = (pos_np[u1] + pos_np[v1] + pos_np[u2] + pos_np[v2]) * 0.25
                    for n in (u1, v1, u2, v2):
                        delta   = pos_np[n] - avg
                        norm    = np.linalg.norm(delta)
                        if norm > 1e-6:
                            pos_np[n] += 1e-2 * (delta / norm)
                        else:
                            rnd = np.random.randn(2)
                            rnd /= np.linalg.norm(rnd)
                            pos_np[n] += 1e-2 * rnd
            # Keep centroid at (0, 0)
            centroid = np.mean(list(pos_np.values()), axis=0)
            for n in pos_np:
                pos_np[n] -= centroid
            if max_overlap < tol:
                break

        # Binary search shrink-to-fit
        def _has_overlap(positions):
            """Return True if any pair of circles overlaps under margin."""
            keys = list(positions)
            for i, j in combinations(keys, 2):
                xi, yi  = positions[i]
                xj, yj  = positions[j]
                req     = self.radius_vals[i] + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE
                if np.hypot(xi - xj, yi - yj) < req - shrink_tol:
                    return True
            return False

        centroid    = np.mean(list(pos_np.values()), axis=0)
        low, high   = 0.0, 1.0
        final_pos   = pos_np.copy()

        while high - low > shrink_tol:
            mid = 0.5 * (low + high)
            # mid=1.0 means no shrink; mid→0 means full collapse
            trial = {n: centroid + mid * (final_pos[n] - centroid)
                    for n in final_pos}
            if _has_overlap(trial):
                high = mid
            else:
                final_pos = trial
                low = mid

        # Write back nanoparticle positions
        for n, coord in final_pos.items():
            self.pos[n] = coord

        # Place electrodes outside forbidden region
        coords_np = np.vstack([final_pos[n] for n in sorted(final_pos)])
        radii_np  = self.radius_vals.copy()
        centroid  = np.mean(coords_np, axis=0)

        # build forbidden region of all existing disks (with padding)
        def _build_forbidden(coords, radii, pad):
            return unary_union([
                Point(x, y).buffer(r + pad, resolution=32)
                for (x, y), r in zip(coords, radii)
            ])

        e_np_pairs  = [(i, -val) for i, val in enumerate(self.net_topology[:,0]) if val >= 0]
        coords_list = coords_np.copy()
        radii_list  = radii_np.copy()

        for i, e_i in e_np_pairs:
            forbidden   = _build_forbidden(coords_list, radii_list, self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE)
            xj, yj      = self.pos[i]
            p_j         = Point(xj, yj)

            # outward‐ray candidate (+ tiny ε to clear buffer artifacts)
            v = np.array([xj, yj]) - centroid
            if np.linalg.norm(v) > 1e-8:
                dir_vec = v / np.linalg.norm(v)
                d       = (radii_list[i] + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE + 1e-6)
                cand    = Point(xj + dir_vec[0]*d, yj + dir_vec[1]*d)
                if forbidden.contains(cand):
                    # fallback to convex hull of forbidden region
                    hull = forbidden.convex_hull
                    _, cand = nearest_points(p_j, hull.boundary)
            else:
                hull = forbidden.convex_hull
                _, cand = nearest_points(p_j, hull.boundary)

            # record electrode position and expand the forbidden set
            self.pos[e_i]   = np.array([cand.x, cand.y])
            coords_list     = np.vstack((coords_list, [cand.x, cand.y]))
            radii_list      = np.hstack((radii_list, self.ELECTRODE_RADIUS))

        # Update distance matrices
        # nanoparticle–nanoparticle distances
        diff = coords_np[:, None, :] - coords_np[None, :, :]
        self.dist_matrix = np.linalg.norm(diff, axis=2)

        # electrode–particle distances
        el_dist = np.zeros((self.N_electrodes, self.N_particles))
        for idx in range(1, self.N_electrodes + 1):
            epos = np.array(self.pos[-idx])
            d = coords_np - epos
            el_dist[idx - 1] = np.linalg.norm(d, axis=1)
        self.electrode_dist_matrix = el_dist

    def pack_for_cubic(self):
        
        r_val  = self.radius_vals[0]
        r_val2 = self.radius_vals[-1]
        
        for n in range(self.N_particles-1):
            self.pos[n] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[n][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[n][1])
        self.pos[self.N_particles-1] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[self.N_particles-1][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*(self.pos[self.N_particles-1][1]-1) + r_val2+r_val + self.MIN_NP_NP_DISTANCE)
        for e in range(1,self.N_electrodes):
            self.pos[-e] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-e][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-e][1])
        self.pos[-self.N_electrodes] = ((2*r_val + self.MIN_NP_NP_DISTANCE)*self.pos[-self.N_electrodes][0],(2*r_val + self.MIN_NP_NP_DISTANCE)*(self.pos[-self.N_electrodes][1]-1) + 2*r_val2 + self.MIN_NP_NP_DISTANCE)

        dist_matrix = np.zeros(shape=(self.N_particles,self.N_particles))
        for i in range(self.N_particles):
            for j in range(self.N_particles):
                dist_matrix[i,j] = np.sqrt((self.pos[i][0]-self.pos[j][0])**2 + (self.pos[i][1]-self.pos[j][1])**2)
        self.dist_matrix = dist_matrix

        el_dist = np.zeros((self.N_electrodes, self.N_particles))
        for i in range(1, self.N_electrodes + 1):
            for j in range(self.N_particles):
                el_dist[i - 1, j] = np.sqrt((self.pos[-i][0]-self.pos[j][0])**2 + (self.pos[-i][1]-self.pos[j][1])**2)
        self.electrode_dist_matrix = el_dist

    def calc_capacitance_matrix(self, eps_r: float = 2.6, eps_s: float = 3.9)->None:
        """
        Calculate the capacitance matrix (NxN) for the nanoparticle network.

        For each nanoparticle:
        - The off-diagonal elements are minus the mutual capacitance to all other nanoparticles.
        - The diagonal element is the sum of all mutual capacitances to other NPs, 
        capacitances to (constant-potential) electrodes, and the nanoparticle's self-capacitance.

        Parameters
        ----------
        eps_r : float, optional
            Relative permittivity of insulating material between nanoparticles (default: 2.6)
        eps_s : float, optional
            Relative permittivity of the environment (e.g., oxide layer) for self-capacitance (default: 3.9)
        
        Raises
        ------
        ValueError
            If permittivity parameters are non-positive.
        RuntimeError
            If radii have not been initialized, or if matrix inversion fails.
        """
        if eps_r <= 0 or eps_s <= 0:
            raise ValueError(f"Permittivities must be positive, got eps_r={eps_r}, eps_s={eps_s}")
        if not hasattr(self, 'radius_vals'):
            raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")
        if not hasattr(self, 'dist_matrix') or not hasattr(self, 'electrode_dist_matrix'):
            raise RuntimeError("Distance matrices not initialized. Call pack_planar_circles first.")

        # Initialize capacitance matrix
        self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
        self.eps_r              = eps_r
        self.eps_s              = eps_s

        # Loop over nanoparticles to fill capacitance matrix
        for i in range(self.N_particles):
            C_sum = 0.0
            # Mutual NP-NP capacitances (off-diagonal)
            for j in range(self.N_particles):
                if i!=j:
                    # If overlap occurs, mutual_capacitance will enforce minimum distance inside itself
                    val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[j], self.dist_matrix[i,j])
                    C_sum   +=  val
                    self.capacitance_matrix[i,j] = -val

            # NP-electrode capacitance (only for constant electrodes)
            for j in range(self.N_electrodes):
                if hasattr(self, 'floating_indices') and j in self.floating_indices:
                    continue  # skip floating electrodes
                val     =   self.mutual_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS, self.electrode_dist_matrix[j,i])
                C_sum   +=  val
                    
            # Self Capacitance
            C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            self.capacitance_matrix[i,i] = C_sum
        
        # Invert capacitance matrix
        try:
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e
        
    def calc_electrode_capacitance_matrix(self):
        """
        Calculate the capacitance matrix between electrodes and nanoparticles.

        For each electrode (constant-potential only), calculates its mutual capacitance
        to every nanoparticle using the specified geometry and physical parameters.

        - Skips floating electrodes (as they are not held at constant potential).
        - Stores result as self.electrode_capacitance_matrix (shape: N_electrodes x N_particles)
        - Stores self-capacitances as self.self_capacitance (length: N_particles)

        Raises
        ------
        RuntimeError
            If required attributes are missing.
        """

        if not hasattr(self, 'N_electrodes') or not hasattr(self, 'N_particles'):
            raise RuntimeError("Network/electrodes not initialized.")
        if not hasattr(self, 'floating_indices'):
            raise RuntimeError("Floating electrode indices not set.")
        if not hasattr(self, 'eps_r') or not hasattr(self, 'eps_s'):
            raise RuntimeError("Physical parameters eps_r, eps_s not set. Call calc_capacitance_matrix first.")
        if not hasattr(self, 'radius_vals') or not hasattr(self, 'electrode_dist_matrix'):
            raise RuntimeError("Required geometry (radius_vals, electrode_dist_matrix) not set. Run packing and init radii.")

        # Boolean mask: True for constant electrodes, False for floating
        constant_mask = np.ones(self.N_electrodes, dtype=bool)
        constant_mask[self.floating_indices] = False

        # C_lead[j, i] = C between electrode j and particle i (zero if floating)
        C_lead = np.zeros((self.N_electrodes, self.N_particles))
        for j in np.nonzero(constant_mask)[0]:
            for i in range(self.N_particles):
                C_lead[j, i] = self.mutual_capacitance_adjacent_spheres(self.eps_r,self.radius_vals[i],
                                                                        self.ELECTRODE_RADIUS,self.electrode_dist_matrix[j, i])

        # Self-capacitances (for reference)
        C_self = np.array([self.self_capacitance_sphere(self.eps_s, r) for r in self.radius_vals])
        self.self_capacitance               = C_self        # shape (N_particles,)
        self.electrode_capacitance_matrix   = C_lead        # shape (N_electrodes, N_particles)
        
    def init_charge_vector(self, voltage_values: np.ndarray) -> None:
        """
        Initialize the nanoparticle charge vector given electrode and gate voltages.

        The resulting vector (self.charge_vector) has shape (N_particles,).
        Convention: `voltage_values` is
            [V_e1, V_e2, ..., V_eN, V_G]
        where N = self.N_electrodes and V_G is the gate voltage.
        - Floating electrode voltages must be zero.

        Parameters
        ----------
        voltage_values : np.ndarray
            1D array of shape (N_electrodes + 1,) with electrode and gate voltages.

        Raises
        ------
        RuntimeError
            If capacitance matrix has not been calculated.
        ValueError
            If input length is wrong or floating electrode voltages are nonzero.
        """
        if not hasattr(self, 'capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {voltage_values.shape[0]}")

        # Check floating electrode voltages (allow tiny floating point errors)
        floating_voltages = voltage_values[self.floating_indices]
        if np.any(np.abs(floating_voltages) > 1e-12):
            raise ValueError("Floating electrode voltages must be initialized to zero")
        
        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        # C_lead.T @ V_e is shape (N_particles,)
        self.charge_vector = self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g

    def get_charge_vector_offset(self, voltage_values : np.ndarray) -> np.ndarray:
        """
        Compute the charge vector offset induced by the provided electrode and gate voltages.

        This method calculates the induced charge on each nanoparticle as a result of changes
        in electrode and gate voltages, without reinitializing the internal charge vector.
        Useful for rapid recalculation when only external voltages change.

        Parameters
        ----------
        voltage_values : np.ndarray
            1D array, shape (N_electrodes + 1,): [V_e1, V_e2, ..., V_eN, V_G] where V_G is the gate.

        Returns
        -------
        np.ndarray
            Charge offset vector (length N_particles) [aC], representing the change in nanoparticle charges.

        Raises
        ------
        ValueError
            If the input array is the wrong shape.
        """
        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {voltage_values.shape[0]}.")

        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        return self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g
    
    # TODO Testing
    def delete_n_junctions(self, n: int) -> None:
        """
        Randomly delete n nanoparticle-nanoparticle junctions (edges), preserving:
        - network connectivity (no bridges/cut-edges are removed)
        - all electrode connectivity

        Parameters
        ----------
        n : int
            Number of nanoparticle-nanoparticle junctions to delete.

        Raises
        ------
        ValueError
            If n is negative or more than the number of deletable (non-bridge) edges.
        """
        if n < 0:
            raise ValueError(f"Number of junctions to delete must be non-negative, got {n}")

        # Extract subgraph of only nanoparticles (no electrodes)
        G_np = self.G.subgraph([i for i in range(self.N_particles)]).to_undirected()

        # Build list of all possible removable edges (no self-loops)
        removable = []
        for i in range(self.N_particles):
            if self.net_topology[i, 0] == self.NO_CONNECTION:
                for j in self.net_topology[i, 1:]:
                    if j != self.NO_CONNECTION and i < int(j):
                        removable.append((i, int(j)))

        # Find bridges (edges whose removal would disconnect the network)
        bridges = set(nx.bridges(G_np))

        # Only allow deletion of non-bridge, non-self-loop edges
        candidates = [edge for edge in removable if edge not in bridges and edge[::-1] not in bridges]
        if n > len(candidates):
            raise ValueError(f"Only {len(candidates)} removable (non-bridge) junctions available, cannot delete {n}.")

        # Randomly pick n edges to delete
        idxs = self.rng.choice(len(candidates), size=n, replace=False)
        to_delete = [candidates[i] for i in idxs]

        for i, j in to_delete:
            # Remove from topology matrix
            i_idx = np.where(self.net_topology[i, 1:] == j)[0]
            j_idx = np.where(self.net_topology[j, 1:] == i)[0]
            if len(i_idx) > 0:
                self.net_topology[i, 1 + i_idx[0]] = self.NO_CONNECTION
            if len(j_idx) > 0:
                self.net_topology[j, 1 + j_idx[0]] = self.NO_CONNECTION
            # Remove from graph (both directions)
            if self.G.has_edge(i, j):
                self.G.remove_edge(i, j)
            if self.G.has_edge(j, i):
                self.G.remove_edge(j, i)

    def get_charge_vector(self) -> np.ndarray:
        """
        Get the current charge vector of the network.

        Returns
        -------
        np.ndarray
            Array of induced charge values for each nanoparticle [aC], shape: (N_particles,)

        Raises
        ------
        RuntimeError
            If charge vector has not been initialized (call init_charge_vector() first).
        """
        if not hasattr(self, 'charge_vector'):
            raise RuntimeError("Charge vector not initialized. Call init_charge_vector first.")
        return self.charge_vector
    
    def get_capacitance_matrix(self) -> np.ndarray:
        """
        Get the network capacitance matrix.

        Returns
        -------
        np.ndarray
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
    
    def get_electrode_capacitance_matrix(self) -> np.ndarray:
        """
        Get the electrode capacitance matrix.

        Returns
        -------
        np.ndarray
            Array containing electrode(lead) capacitance values [aF]
            Shape: (N_electrodes, N_particles)
            
        Raises
        ------
        RuntimeError
            If electrode capacitance matrix hasn't been calculated
        """
        if not hasattr(self, 'electrode_capacitance_matrix'):
            raise RuntimeError("Capacitance matrix not calculated. Call calc_electrode_capacitance_matrix first.")
        return self.electrode_capacitance_matrix
    
    def get_self_capacitance(self) -> np.ndarray:
        """Get the self capacitance of each NP.

        Returns
        -------
        np.ndarray
            Array containing self capacitance values [aF]
            Shape: (N_particles,)
            
        Raises
        ------
        RuntimeError
            If self capacitance values hasn't been calculated
        """
        if not hasattr(self, 'self_capacitance'):
            raise RuntimeError("Self Capacitance not calculated. Call calc_electrode_capacitance_matrix first.")
        return self.self_capacitance
        
    def get_inv_capacitance_matrix(self) -> np.ndarray:
        """Get the inverse capacitance matrix.

        Returns
        -------
        np.ndarray
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

    def get_dist_matrix(self) -> np.ndarray:
        """Get the network distance matrix

        Returns
        -------
        np.ndarray
            Array containing network distances [nm]
            Shape: (N_particles, N_particles)
        
        Raises
        ------
        RuntimeError
            If distance matrix hasn't been calculated
        """
        if not hasattr(self, 'dist_matrix'):
            raise RuntimeError("Distance matrix not calculated. Call pack_planar_circles first.")
        return self.dist_matrix
    
    def get_electrode_dist_matrix(self) -> np.ndarray:
        """Get the electrode distance matrix

        Returns
        -------
        np.ndarray
            Array containing network distances [nm]
            Shape: (N_electrodes, N_particles)
        
        Raises
        ------
        RuntimeError
            If electrode distance matrix hasn't been calculated
        """
        if not hasattr(self, 'electrode_dist_matrix'):
            raise RuntimeError("Electrode distance matrix not calculated. Call pack_planar_circles first.")
        return self.electrode_dist_matrix
    
    def get_radius(self) -> np.ndarray:
        """Get the radius of each NP

        Returns
        -------
        np.ndarray
            Array containing nanoparticle radius [nm]
            Shape: (N_particles,)
        
        Raises
        ------
        RuntimeError
            If radius hasn't been calculated
        """
        if not hasattr(self, "radius_vals"):
            raise RuntimeError("Nanoparticle radius not defined. Call init_nanoparticle_radius first.")
        return self.radius_vals
    
    # def calc_capacitance_matrix(self, eps_r: float = 2.6, eps_s: float = 3.9)->None:
    #     """
    #     Calculate the capacitance matrix of the nanoparticle network.

    #     Parameters
    #     ----------
    #     eps_r : float
    #         Relative permittivity of insulating material between nanoparticles
    #     eps_s : float
    #         Relative permittivity of insulating environment (oxide layer)
            
    #     Raises
    #     ------
    #     ValueError
    #         If physical parameters are invalid
    #     RuntimeError
    #         If matrix inversion fails
    #     """
    #     if not hasattr(self, 'radius_vals'):
    #         raise RuntimeError("Nanoparticle radii not initialized. Call init_nanoparticle_radius first.")

    #     # Initialize capacitance matrix
    #     self.capacitance_matrix = np.zeros((self.N_particles,self.N_particles))
    #     self.eps_r              = eps_r
    #     self.eps_s              = eps_s

    #     # Loop over each particle to calculate capacitance contributions
    #     for i in range(self.N_particles):
    #         C_sum = 0.0
    #         # Iterate over the junctions of the current nanoparticle
    #         for j in range(self.N_junctions+1):
    #             neighbor = self.net_topology[i,j]

    #             # Skip if the neighbor is invalid
    #             if neighbor == self.NO_CONNECTION:
    #                 continue

    #             # Capacitance with the electrode (j == 0)
    #             if (j == 0):
    #                 if neighbor-1 not in self.floating_indices:
    #                     # Add electrode capacitance (using mutual capacitance formula)
    #                     C_sum += self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)

    #             else:
    #                 # Calc mutual capacitance
    #                 val = self.mutal_capacitance_adjacent_spheres(eps_r, self.radius_vals[i], self.radius_vals[int(neighbor)])
    #                 self.capacitance_matrix[i,int(neighbor)] = -val
    #                 C_sum += val

    #         # Self Capacitance
    #         C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            
    #         # Set diagonal element (total capacitance)
    #         self.capacitance_matrix[i,i] = C_sum
        
    #     # Calculate the inverse capacitance matrix
    #     try:
    #         self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
    #     except np.linalg.LinAlgError as e:
    #         raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e

    # def init_charge_vector(self, voltage_values: np.array)->None:
    #     """
    #     Initialize the charge vector based on electrode voltages.

    #     Parameters
    #     ----------
    #     voltage_values : array
    #        Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
    #        where V_G is the gate voltage

    #     Raises
    #     ------
    #     ValueError
    #         If voltage array length doesn't match number of electrodes + 1
    #         If floating electrode voltages are non-zero
    #     RuntimeError
    #         If capacitance matrix hasn't been calculated
    #     """
    #     if not hasattr(self, 'capacitance_matrix'):
    #         raise RuntimeError("Capacitance matrix not calculated. Call calc_capacitance_matrix first.")

    #     if len(voltage_values) != self.N_electrodes + 1:
    #         raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {len(voltage_values)}")

    #     floating_voltages = [voltage_values[i] for i in self.floating_indices]
    #     if any(v != 0 for v in floating_voltages):
    #         raise ValueError("Floating electrode voltages must be initialized to zero")

    #     # Initialize charge vector
    #     self.charge_vector = np.zeros(self.N_particles)

    #     # Iterate over all nanoparticles
    #     for i in range(self.N_particles):
    #         electrode_index = int(self.net_topology[i,0] - 1)

    #         if self.net_topology[i,0] != self.NO_CONNECTION:
    #             # If connected to an electrode, calculate charge from electrode voltage
    #             if electrode_index not in self.floating_indices:
    #                 C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
    #             else:
    #                 C_lead  = 0.0
                
    #             C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             self.charge_vector[i] = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
    #         else:
    #             # If not connected to an electrode
    #             C_self = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             self.charge_vector[i] = voltage_values[-1] * C_self

    # def get_charge_vector_offset(self, voltage_values : np.array)->np.array:
    #     """
    #     Calculate the charge vector offset induced by electrode voltages.
        
    #     This method computes the charges induced on nanoparticles by the electrodes
    #     without reinitializing the entire charge vector. This is useful when you want
    #     to update only the portion of charges that changes due to external voltage changes.

    #     Parameters
    #     ----------
    #     voltage_values : np.array
    #         Electrode voltages as np.array([V_e1, V_e2, V_e3, ..., V_G])
    #         where V_G is the gate voltage

    #     Returns
    #     -------
    #     np.array
    #         Charge offset vector [aC] representing charges induced by electrodes
    #     """

    #     offset = np.zeros(self.N_particles)

    #     # For each charge
    #     for i in range(self.N_particles):
    #         electrode_index = int(self.net_topology[i,0] - 1)

    #         if self.net_topology[i,0] != self.NO_CONNECTION:
    #             # If connected to an electrode, calculate charge from electrode voltage
    #             if electrode_index not in self.floating_indices:
    #                 C_lead  = self.mutal_capacitance_adjacent_spheres(self.eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS)
    #             else:
    #                 C_lead  = 0.0
                
    #             C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             offset[i]   = voltage_values[electrode_index] * C_lead + voltage_values[-1] * C_self
            
    #         else:
    #             # If not connected to an electrode
    #             C_self      = self.self_capacitance_sphere(self.eps_s, self.radius_vals[i])
    #             offset[i]   = voltage_values[-1] * C_self

    #     return offset