import numpy as np
import networkx as nx
import topology
from typing             import List
from scipy.optimize     import least_squares
from scipy.spatial      import KDTree
from shapely.geometry   import Point, Polygon, LineString
from shapely.ops        import unary_union, nearest_points
from itertools          import combinations

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

    def mutal_capacitance_adjacent_spheres_sinh(self, eps_r: float, np_radius1: float, np_radius2: float, distance: float, N_sum: int = 50) -> float:
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
        distance : float
            Center-to-Center distance [nm]
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
        if distance <= 0:
            raise ValueError(f"Distance must be positive, got {distance}")
        
        # Enforce minimum separation
        if distance < (np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE):
            distance = np_radius1 + np_radius2 + self.MIN_NP_NP_DISTANCE

        # Base factor
        factor  = 4 * self.PI * self.EPSILON_0 * eps_r * (np_radius1 * np_radius2) / distance
        arg     = (distance**2 - np_radius1**2 - np_radius2**2) / (2*np_radius1*np_radius2)
        U_val   = np.arccosh(arg)
        s       = np.sum([1.0/np.sinh(n*U_val) for n in range(1,N_sum+1)])
        cap     = factor * np.sinh(U_val) * s

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
    
    def pack_planar_circles(self, max_iter: int = 500, alpha: float = 0.05, tol: float = 1e-3, shrink_tol: float = 1e-6) -> None:
        """
        Adjusts `self.pos` so that:
        1. All nanoparticle circles (of varying radii) are non-overlapping.
        2. All graph edges remain straight and non-crossing (planar).
        3. The final packing is as dense as possible (global shrink-to-fit).
        4. Electrode spheres are then placed outside without overlap.

        Parameters
        ----------
        max_iter : int, optional
            Maximum number of local repulsion/attraction iterations.
        alpha : float, optional
            Strength of edge-based attraction (0 < alpha < 1).
        tol : float, optional
            Tolerance for treating two circles as overlapping.
        shrink_tol : float, optional
            Precision for the final binary‐search shrink-to-fit.

        Notes
        -----
        - `self.G` must already be a planar NetworkX graph.
        - `self.pos` is a dict {node_index: np.array([x, y])}.
        - `self.radius_vals` is an array of length N_particles.
        - `self.MIN_NP_NP_DISTANCE` is the required clearance margin.
        - Electrode nodes are labeled -1, -2, …, -N_electrodes and stripped out
        during the nanoparticle packing phase.
        """
        # -------------------------------------------------------------------------
        # 1. Extract the subgraph of nanoparticles only, with their initial positions
        # -------------------------------------------------------------------------
        G_np = self.G.copy()
        for e in range(1, self.N_electrodes + 1):
            G_np.remove_node(-e)
        pos_np = {n: np.array(self.pos[n], dtype=float) for n in self.pos if n >= 0}
        
        # -------------------------------------------------------------------------
        # 2. Uniformly scale edges so each is at least (r_u + r_v + margin) long
        # -------------------------------------------------------------------------
        min_scale = 1.0
        for u, v in G_np.edges():
            p_u, p_v = pos_np[u], pos_np[v]
            dist = np.linalg.norm(p_u - p_v)
            required = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
            if dist > 0:
                min_scale = max(min_scale, required / dist)
        if min_scale > 1.0:
            for n in pos_np:
                pos_np[n] *= min_scale

        # -------------------------------------------------------------------------
        # 3. Local relaxation: repel overlaps + attract neighbors + fix crossings
        # -------------------------------------------------------------------------
        for _ in range(max_iter):
        
            max_overlap = 0.0

            # A) Repel any overlapping circles
            for u, v in combinations(G_np.nodes(), 2):
                p_u, p_v = pos_np[u], pos_np[v]
                req = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
                delta = p_u - p_v
                d = np.hypot(*delta)
                overlap = req - d
                if overlap > tol:
                    max_overlap = max(max_overlap, overlap)
                    # compute unit direction (random if coincident)
                    if d < 1e-6:
                        direction = np.random.randn(2)
                        direction /= np.linalg.norm(direction)
                    else:
                        direction = delta / d
                    shift = 0.5 * overlap * direction
                    pos_np[u] +=  shift
                    pos_np[v] -=  shift

            # B) Attract along each graph edge if too far
            for u, v in G_np.edges():
                delta = pos_np[v] - pos_np[u]
                d = np.hypot(*delta)
                desired = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
                if d > desired + tol:
                    shift_amount = alpha * (d - desired)
                    direction = delta / d
                    pos_np[u] +=  shift_amount * direction
                    pos_np[v] -=  shift_amount * direction

            # C) Fix any crossing edges (should be rare)
            for (u1, v1), (u2, v2) in combinations(G_np.edges(), 2):
                if len({u1, v1, u2, v2}) < 4:
                    continue
                seg1 = LineString([pos_np[u1], pos_np[v1]])
                seg2 = LineString([pos_np[u2], pos_np[v2]])
                if seg1.crosses(seg2):
                    avg = (pos_np[u1] + pos_np[v1] + pos_np[u2] + pos_np[v2]) * 0.25
                    for n in (u1, v1, u2, v2):
                        delta = pos_np[n] - avg
                        norm = np.linalg.norm(delta)
                        if norm > 1e-6:
                            pos_np[n] += 1e-2 * (delta / norm)
                        else:
                            rnd = np.random.randn(2)
                            rnd /= np.linalg.norm(rnd)
                            pos_np[n] += 1e-2 * rnd
            
            centroid = np.mean(list(pos_np.values()), axis=0)
            for n in pos_np:
                pos_np[n] -= centroid
            
            if max_overlap < tol:
                break

        # -------------------------------------------------------------------------
        # 4. Global shrink-to-fit: binary search the largest scale with no overlap
        # -------------------------------------------------------------------------
        def _has_overlap(positions):
            """Return True if any pair of circles overlaps under margin."""
            keys = list(positions)
            for i, j in combinations(keys, 2):
                xi, yi = positions[i]
                xj, yj = positions[j]
                req = self.radius_vals[i] + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE
                if np.hypot(xi - xj, yi - yj) < req - shrink_tol:
                    return True
            return False

        centroid = np.mean(list(pos_np.values()), axis=0)
        low, high = 0.0, 1.0
        final_pos = pos_np.copy()

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

        # -------------------------------------------------------------------------
        # 5. Now place electrodes at the “corners” without overlap
        # -------------------------------------------------------------------------
        coords_np = np.vstack([final_pos[n] for n in sorted(final_pos)])
        radii_np  = self.radius_vals.copy()
        centroid  = np.mean(coords_np, axis=0)

        # build forbidden region of all existing disks (with padding)
        def _build_forbidden(coords, radii, pad):
            return unary_union([
                Point(x, y).buffer(r + pad, resolution=32)
                for (x, y), r in zip(coords, radii)
            ])

        e_np_pairs = [(i, -val) for i, val in enumerate(self.net_topology[:,0]) if val >= 0]
        coords_list = coords_np.copy()
        radii_list  = radii_np.copy()

        for i, e_i in e_np_pairs:
            forbidden = _build_forbidden(coords_list, radii_list,
                                        self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE)
            xj, yj = self.pos[i]
            p_j = Point(xj, yj)

            # outward‐ray candidate (+ tiny ε to clear buffer artifacts)
            v = np.array([xj, yj]) - centroid
            if np.linalg.norm(v) > 1e-8:
                dir_vec = v / np.linalg.norm(v)
                d = (radii_list[i]
                    + self.ELECTRODE_RADIUS
                    + self.MIN_NP_NP_DISTANCE
                    + 1e-6)
                cand = Point(xj + dir_vec[0]*d, yj + dir_vec[1]*d)
                if forbidden.contains(cand):
                    # fallback to convex hull of forbidden region
                    hull = forbidden.convex_hull
                    _, cand = nearest_points(p_j, hull.boundary)
            else:
                hull = forbidden.convex_hull
                _, cand = nearest_points(p_j, hull.boundary)

            # record electrode position and expand the forbidden set
            self.pos[e_i] = np.array([cand.x, cand.y])
            coords_list = np.vstack((coords_list, [cand.x, cand.y]))
            radii_list  = np.hstack((radii_list, self.ELECTRODE_RADIUS))

        # -------------------------------------------------------------------------
        # 6. Update distance matrices for downstream use
        # -------------------------------------------------------------------------
        # nanoparticle–nanoparticle distances
        diff = coords_np[:, None, :] - coords_np[None, :, :]
        self.dist_matrix = np.linalg.norm(diff, axis=2)

        # electrode–particle distances
        el_dist = np.zeros((self.N_electrodes, self.N_particles))
        for idx in range(1, self.N_electrodes + 1):
            epos = np.array(self.pos[-idx])
            d   = coords_np - epos
            el_dist[idx - 1] = np.linalg.norm(d, axis=1)
        self.electrode_dist_matrix = el_dist

    # def pack_planar_circles(self, max_iter=500, alpha = 0.05, tol=1e-3, shrink_tol=1e-6) -> None:
        
    #     # DEFINE NANOPARTICLE POSITIONS
    #     ###############################
        
    #     # Remove electrode nodes
    #     G_old   = self.G.copy()
    #     pos_old = self.pos.copy()
    #     for i in range(1, self.N_electrodes+1):
    #         G_old.remove_node(-i)
    #         del pos_old[-i]
        
    #     # 1) Check planarity & get a planar embedding
    #     is_planar, embedding = nx.check_planarity(G_old, False)
    #     if not is_planar:
    #         raise ValueError("Graph is not planar!")
        
    #     # 3) Scale up so that every edge is at least radii[u]+radii[v]+margin long
    #     min_scale = 0.0
    #     for u, v in G_old.edges():
    #         d           = np.linalg.norm(np.array(pos_old[u]) - np.array(pos_old[v]))
    #         req         = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
    #         min_scale   = max(min_scale, req / d)
            
    #     if min_scale > 1.0:
    #         for n in pos_old:
    #             pos_old[n] = np.array(pos_old[n]) * min_scale

    #     # 4) Iteratively resolve any circle–circle overlaps
    #     for _ in range(max_iter):
    #         max_overlap = 0.0

    #         # A) build shapely circles & check overlaps
    #         circles = {n: Point(*pos_old[n]).buffer(self.radius_vals[n]) for n in G_old}
    #         for u in G_old:
    #             cu = circles[u]
    #             for v in G_old:
    #                 if v <= u: 
    #                     continue
    #                 cv = circles[v]
    #                 if cu.intersects(cv):
    #                     # how much overlap?
    #                     dist = cu.centroid.distance(cv.centroid)
    #                     overlap = (self.radius_vals[u] + self.radius_vals[v]) - dist + self.MIN_NP_NP_DISTANCE
    #                     max_overlap = max(max_overlap, overlap)
    #                     # push each one half the overlap in opposite directions
    #                     dir_vec = (pos_old[u] - pos_old[v])
    #                     norm = np.linalg.norm(dir_vec)
    #                     if norm < 1e-6:
    #                         # random tiny jitter if they coincide
    #                         dir_vec = np.random.randn(2)
    #                         norm = np.linalg.norm(dir_vec)
    #                     dir_vec /= norm
    #                     shift = 0.5 * overlap * dir_vec
    #                     pos_old[u] += shift
    #                     pos_old[v] -= shift
            
    #         for u, v in G_old.edges():
    #             dx, dy = pos_old[v] - pos_old[u]
    #             d = np.hypot(dx, dy)
    #             desired = self.radius_vals[u] + self.radius_vals[v] + self.MIN_NP_NP_DISTANCE
    #             if d > desired + tol:
    #                 # pull them each α×(d-desired) closer
    #                 shift = alpha * (d - desired)
    #                 ux, uy = dx/d, dy/d
    #                 pos_old[u] +=  shift * np.array([ux, uy])
    #                 pos_old[v] -=  shift * np.array([ux, uy])

    #         # B) also check for any crossing edges (they should never happen,
    #         #     but numerical pushes might induce one).  If you find a crossing,
    #         #     nudge the four endpoints apart similarly.
    #         for (u1, v1), (u2, v2) in combinations(G_old.edges(), 2):
    #             # skip sharing a node
    #             if len({u1,v1,u2,v2}) < 4:
    #                 continue
    #             seg1 = LineString([pos_old[u1], pos_old[v1]])
    #             seg2 = LineString([pos_old[u2], pos_old[v2]])
    #             if seg1.crosses(seg2):
    #                 # nudge endpoints away along the bisector of segments
    #                 # (simple hack—pull each endpoint slightly off)
    #                 for n in (u1, v1, u2, v2):
    #                     pos_old[n] += 1e-2 * (pos_old[n] - (pos_old[u1]+pos_old[v1]+pos_old[u2]+pos_old[v2])/4)

    #         if max_overlap < tol:
    #             break
        
    #     # After your local relaxation converges, `pos_old` is overlap-free
    #     # but may have voids. Now do a global shrink-to-fit:
    #     def has_any_overlap(positions):
    #         keys = list(positions)
    #         for a in range(len(keys)):
    #             i = keys[a]
    #             xi, yi = positions[i]
    #             for b in range(a+1, len(keys)):
    #                 j = keys[b]
    #                 xj, yj = positions[j]
    #                 req = self.radius_vals[i] + self.radius_vals[j] + self.MIN_NP_NP_DISTANCE
    #                 if np.hypot(xi - xj, yi - yj) < req - shrink_tol:
    #                     return True
    #         return False

    #     # centroid of current packing
    #     centroid = np.mean([pos_old[n] for n in pos_old], axis=0)

    #     # binary‐search shrink factor between 1.0 and 0.0
    #     low, high = 0.0, 1.0
    #     final_pos = pos_old.copy()

    #     while high - low > shrink_tol:
    #         mid = 0.5 * (low + high)
    #         factor = 1.0 - (1.0 - mid)  # can adjust to control shrink speed
    #         # propose scaled positions
    #         trial = {
    #             n: centroid + factor * (final_pos[n] - centroid)
    #             for n in final_pos
    #         }
    #         if has_any_overlap(trial):
    #             high = mid
    #         else:
    #             final_pos = trial
    #             low = mid

    #     # Update nanoparticle positions
    #     coords = np.array([val for _, val in pos_old.items()])
    #     for i in range(self.N_particles):
    #         self.pos[i] = pos_old[i]

    #     # DEFINE ELECTRODE POSITIONS
    #     ############################
    #     net_topo    = self.net_topology.copy()
    #     e_np_pairs  = [(i,-val) for i, val in enumerate(net_topo[:,0]) if val >= 0]
    #     coords_new  = coords.copy()
    #     radii_new   = self.radius_vals.copy()
    #     centroid    = np.mean(coords_new, axis=0)

    #     for i, e_i in e_np_pairs:

    #         # Build the forbidden region = union of all existing circles grown by r_s
    #         forbidden = unary_union([
    #             Point(x_i, y_i).buffer(
    #                 r_i + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE
    #                 )
    #                 for (x_i, y_i), r_i in zip(coords_new, radii_new)])

    #         # unpack target
    #         x_j, y_j    = self.pos[i][0], self.pos[i][1]
    #         p_j         = Point(x_j, y_j)

    #         # Outward-Ray placement
    #         v = np.array([x_j,y_j]) - centroid
    #         if np.linalg.norm(v) > 1e-8:
    #             dir_vec = v / np.linalg.norm(v)
    #             # place it exactly at the grown‐circle distance along that ray
    #             cand = Point(
    #                 x_j + dir_vec[0] * (radii_new[i] + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE + 1e-6),
    #                 y_j + dir_vec[1] * (radii_new[i] + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE + 1e-6))
                
    #             # check it’s still in the feasible set
    #             if forbidden.contains(cand):
    #                 hull = forbidden.convex_hull
    #                 _, cand = nearest_points(p_j, hull.boundary)
    #                 # _, cand = nearest_points(p_j, forbidden.boundary)
    #         else:
    #             hull = forbidden.convex_hull
    #             _, cand = nearest_points(p_j, hull.boundary)
    #             # _, cand = nearest_points(p_j, forbidden.boundary)

    #         self.pos[e_i]   = (cand.x, cand.y)
    #         coords_new      = np.vstack((coords_new,(cand.x, cand.y)))
    #         radii_new       = np.hstack((radii_new,self.ELECTRODE_RADIUS))
        
    #     # Nanoparticle distance matrix
    #     diff                = coords[:, None, :] - coords[None, :, :]
    #     dist_matrix         = np.linalg.norm(diff, axis=2)
    #     self.dist_matrix    = dist_matrix

    #     # Electrode-particle distance matrix
    #     el_dist = np.zeros((self.N_electrodes, self.N_particles))
    #     for i in range(1, self.N_electrodes+1):
    #         epos            = np.array(self.pos[-i])
    #         d               = coords - epos
    #         el_dist[i-1]    = np.linalg.norm(d, axis=1)
    #     self.electrode_dist_matrix = el_dist
                
    def pack_planar_circles_dense(self, w_neigh: float = 1e0, w_minsep: float = 1e3, w_planar: float = 1e5, margin: float = 1e-3) -> None:
        """
        Adjust positions of nanoparticles and electrodes to enforce minimum separation and planarity.

        This method performs a two‐step optimization:
        1. A spring‐hinge solver adjusts nanoparticle positions so that:
        - Neighboring particles are pulled to exactly their minimal gap (radius_i + radius_j + 1).
        - All particle pairs maintain at least minimal separation (radius_i + radius_j + 1).
        - Non‐adjacent edges remain non‐crossing (planarity constraint).
        2. Electrodes are repositioned based on lattice direction vectors so that each electrode
        sits at a fixed gap from its connected nanoparticle.

        Parameters
        ----------
        w_neigh : float, optional
            Weight of the spring force pulling neighbors to their target gap (default: 1.0).
        w_minsep : float, optional
            Weight of the hinge penalty enforcing minimal separation for all pairs (default: 10.0).
        w_planar : float, optional
            Weight of the hinge penalty forbidding edge crossings (default: 1e5).
        margin : float, optional
            Minimal clearance between non‐adjacent edges before planarity penalty activates (default: 0.0).
        """
        # Get NP-Electrode pairs
        net_topo    = self.net_topology.copy()
        pos_old     = self.pos.copy()
        e_np_pairs  = [(i,-val) for i, val in enumerate(net_topo[:,0]) if val >= 0]

        # Copy attributes
        radii       = self.radius_vals.copy()
        nanop_adj   = self.net_topology[:,1:]
        radii_sort  = np.sort(radii)
        max_spacing = radii_sort[-1]+radii_sort[-2]+self.MIN_NP_NP_DISTANCE

        # Build adjacency mask for nanoparticles
        neighbors   = [np.unique(nanop_adj[i][nanop_adj[i]>=0].astype(int)) for i in range(self.N_particles)]
        max_deg     = max((len(ne) for ne in neighbors), default=0)
        adj_mask    = -np.ones((self.N_particles, max_deg), dtype=int)
        for i, ne in enumerate(neighbors):
            adj_mask[i,:len(ne)] = ne

        # Initial positions
        x0 = np.zeros((self.N_particles,2))
        for i in range(self.N_particles):
            x0[i] = pos_old.get(i, (0.0,0.0))
        x0 = x0*max_spacing
         
        # Perform spring/hinge solver
        coords = self._pack_down(adj_mask, radii, w_neigh, w_minsep, w_planar, margin, x0.ravel())

        # Update nanoparticle positions
        for i in range(self.N_particles):
            self.pos[i] = tuple(coords[i])

        # DEFINE ELECTRODE POSITIONS
        ############################

        coords_new  = coords.copy()
        radii_new   = radii.copy()
        centroid    = np.mean(coords_new, axis=0)

        for i, e_i in e_np_pairs:

            # Build the forbidden region = union of all existing circles grown by r_s
            forbidden = unary_union([
                Point(x_i, y_i).buffer(
                    r_i + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE
                    )
                    for (x_i, y_i), r_i in zip(coords_new, radii_new)])

            # unpack target
            x_j, y_j    = self.pos[i][0], self.pos[i][1]
            p_j         = Point(x_j, y_j)

            # Outward-Ray placement
            v = np.array([x_j,y_j]) - centroid
            if np.linalg.norm(v) > 1e-8:
                dir_vec = v / np.linalg.norm(v)
                # place it exactly at the grown‐circle distance along that ray
                cand = Point(
                    x_j + dir_vec[0] * (radii_new[i] + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE + 1e-6),
                    y_j + dir_vec[1] * (radii_new[i] + self.ELECTRODE_RADIUS + self.MIN_NP_NP_DISTANCE + 1e-6))
                
                # check it’s still in the feasible set
                if forbidden.contains(cand):
                    hull = forbidden.convex_hull
                    _, cand = nearest_points(p_j, hull.boundary)
                    # _, cand = nearest_points(p_j, forbidden.boundary)
            else:
                hull = forbidden.convex_hull
                _, cand = nearest_points(p_j, hull.boundary)
                # _, cand = nearest_points(p_j, forbidden.boundary)

            self.pos[e_i]   = (cand.x, cand.y)
            coords_new      = np.vstack((coords_new,(cand.x, cand.y)))
            radii_new       = np.hstack((radii_new,self.ELECTRODE_RADIUS))
        
        # Nanoparticle distance matrix
        diff                = coords[:, None, :] - coords[None, :, :]
        dist_matrix         = np.linalg.norm(diff, axis=2)
        self.dist_matrix    = dist_matrix

        # Electrode-particle distance matrix
        el_dist = np.zeros((self.N_electrodes, self.N_particles))
        for i in range(1, self.N_electrodes+1):
            epos            = np.array(self.pos[-i])
            d               = coords - epos
            el_dist[i-1]    = np.linalg.norm(d, axis=1)
        self.electrode_dist_matrix = el_dist

    def _pack_down(self,adj_array: np.ndarray, radii: np.ndarray, w_neigh: float,
                   w_minsep: float, w_planar: float, margin: float, x0: np.ndarray) -> np.ndarray:
        """
        Pack‑down solver for all nodes, using an initial position guess.

        This internal helper runs a nonlinear least‑squares solve that:
        - Pulls each neighbor pair (i,j) to exactly distance radii[i]+radii[j]+1
            (spring residuals weighted by `w_neigh`),
        - Enforces a global minimum center‑to‑center gap of radii[i]+radii[j]+1
            for *all* pairs (hinge residuals weighted by `w_minsep`),
        - Prevents any edge crossings via hinge penalties on segment–segment
            distances (weighted by `w_planar`, activating when < `margin`).

        Parameters
        ----------
        adj_array : np.ndarray, shape (N, M)
            Masked adjacency list: each row i lists its neighbor indices (mask non‑neighbors <0).
        radii : np.ndarray, shape (N,)
            Radii of the N circles/nodes.
        w_neigh : float
            Spring weight for neighbor‑distance residuals.
        w_minsep : float
            Hinge‑loss weight enforcing global non‑overlap.
        w_planar : float
            Hinge‑loss weight forbidding edge crossings.
        margin : float
            Clearance threshold before planarity penalty kicks in.
        x0 : np.ndarray, shape (2*N,)
            Initial guess for the flattened (x,y) coordinates of all N nodes.

        Returns
        -------
        coords : np.ndarray, shape (N, 2)
            Optimized 2D coordinates of each node center.
        """
        N = len(radii)
        adj = np.asarray(adj_array)

        # 1) Build neighbor list and undirected edges
        neighbors = [adj[i][adj[i]>=0].astype(int) for i in range(N)]
        edges     = [(i,j) for i in range(N) for j in neighbors[i] if j>i]

        # 2) Precompute target gap for every pair once
        min_gap = { (i,j): radii[i] + radii[j] + 1 for i in range(N) for j in range(i+1,N) }

        # 3) Prune the global-overlap pairs with a KDTree on x0
        coords0 = x0.reshape(N,2)
        tree    = KDTree(coords0)
        max_gap = max(min_gap.values())
        # any pair whose initial distance > max_gap + margin cannot violate the hinge
        overlap_pairs = []
        for i in range(N):
            for j in tree.query_ball_point(coords0[i], r=max_gap):
                if j>i:
                    overlap_pairs.append((i,j))

        # 4) Prune edge-pairs for planarity by bounding‐box overlap on x0
        #    Build bbox centers & half‐diagonals
        mids    = [(coords0[i]+coords0[j])/2 for i,j in edges]
        lengths = [np.linalg.norm(coords0[i]-coords0[j]) for i,j in edges]
        # radius of each edge's bbox circle
        bbox_r  = [L/2 + margin for L in lengths]
        ed_tree = KDTree(mids)
        planarity_pairs = []
        for idx,(a,b) in enumerate(edges):
            # find candidate edges whose midpoints lie within sum of radii
            for cidx in ed_tree.query_ball_point(mids[idx], r=bbox_r[idx]+max(bbox_r)/2):
                if cidx>idx:
                    c,d = edges[cidx]
                    # ensure disjoint
                    if {a,b}.isdisjoint({c,d}):
                        planarity_pairs.append((a,b,c,d))

        def residuals(x):
            X = x.reshape(N,2)
            res = []

            # 5a) Neighbor springs
            for (i,j) in edges:
                d = np.linalg.norm(X[i]-X[j])
                res.append(w_neigh * (d - min_gap[(i,j)]))

            # 5b) Overlap hinges (pruned)
            for (i,j) in overlap_pairs:
                d = np.linalg.norm(X[i]-X[j])
                pen = max(0.0, min_gap[(i,j)] - d)
                res.append(w_minsep * pen)

            # 5c) Planarity hinges (pruned)
            for (i,j,k,l) in planarity_pairs:
                dseg = self._seg_seg_dist(X[i], X[j], X[k], X[l])
                pen  = max(0.0, margin - dseg)
                res.append(w_planar * pen)

            return np.array(res)

        # 6) Solve
        sol = least_squares(residuals, x0, method='trf')
        return sol.x.reshape(N, 2)

    # def _pack_down(self, adj_array: np.ndarray, radii: np.ndarray, w_neigh: float, w_minsep: float,
    #                         w_planar: float, margin: float, x0: np.ndarray) -> np.ndarray:
    #     """
    #     Pack‑down solver for all nodes, using an initial position guess.

    #     This internal helper runs a nonlinear least‑squares solve that:
    #     - Pulls each neighbor pair (i,j) to exactly distance radii[i]+radii[j]+1
    #         (spring residuals weighted by `w_neigh`),
    #     - Enforces a global minimum center‑to‑center gap of radii[i]+radii[j]+1
    #         for *all* pairs (hinge residuals weighted by `w_minsep`),
    #     - Prevents any edge crossings via hinge penalties on segment–segment
    #         distances (weighted by `w_planar`, activating when < `margin`).

    #     Parameters
    #     ----------
    #     adj_array : np.ndarray, shape (N, M)
    #         Masked adjacency list: each row i lists its neighbor indices (mask non‑neighbors <0).
    #     radii : np.ndarray, shape (N,)
    #         Radii of the N circles/nodes.
    #     w_neigh : float
    #         Spring weight for neighbor‑distance residuals.
    #     w_minsep : float
    #         Hinge‑loss weight enforcing global non‑overlap.
    #     w_planar : float
    #         Hinge‑loss weight forbidding edge crossings.
    #     margin : float
    #         Clearance threshold before planarity penalty kicks in.
    #     x0 : np.ndarray, shape (2*N,)
    #         Initial guess for the flattened (x,y) coordinates of all N nodes.

    #     Returns
    #     -------
    #     coords : np.ndarray, shape (N, 2)
    #         Optimized 2D coordinates of each node center.
    #     """
    #     N           = len(radii)
    #     adj         = np.asarray(adj_array)

    #     # Build neighbor list and undirected edges
    #     neighbors   = [adj[i][adj[i]>=0].astype(int) for i in range(N)]
    #     edges       = [(i,j) for i in range(N) for j in neighbors[i] if j>i]

    #     # Precompute all unique node‐pairs and edge‐pairs for planarity
    #     all_pairs   = [(i,j) for i in range(N) for j in range(i+1,N)]
    #     edge_pairs  = [(a,b,c,d)
    #                     for (a,b) in edges for (c,d) in edges
    #                     if {a,b}.isdisjoint({c,d}) and a<b and c<d]
        
    #     # Minimal allowed distances for every pair (neighbor‐spring target and global separation)
    #     min_all     = {(i,j): radii[i]+radii[j]+1 for (i,j) in all_pairs}

    #     def residuals(x):
    #         X = x.reshape(N,2)
    #         res = []

    #         # Spring residual for each neighbor: drive d -> exact min_all
    #         for (i,j) in edges:
    #             d = np.linalg.norm(X[i]-X[j])
    #             res.append(w_neigh*(d - min_all[(i,j)]))

    #         # Hinge residual for global non-overlap: penalize if d < min_all
    #         for (i,j) in all_pairs:
    #             d = np.linalg.norm(X[i]-X[j]); pen = max(0.0, min_all[(i,j)]-d)
    #             res.append(w_minsep*pen)
            
    #         # Planarity hinge: penalize if any two non-adjacent edges come closer than margin
    #         for (i,j,k,l) in edge_pairs:
    #             dseg = self._seg_seg_dist(X[i],X[j],X[k],X[l]); pen = max(0.0, margin-dseg)
    #             res.append(w_planar*pen)

    #         return np.array(res)
        
    #     # Run the least-squares solver (using x0 as the starting point)
    #     sol = least_squares(residuals, x0, method='trf')

    #     return sol.x.reshape(N,2)
    
    @staticmethod                    
    def _seg_seg_dist(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute shortest distance between two line segments in 2D.

        The distance is the minimum Euclidean distance between any point on segment P1->P2
        and any point on segment Q1->Q2.

        Parameters
        ----------
        p1 : np.ndarray, shape (2,)
            Coordinates of the first endpoint of segment 1.
        p2 : np.ndarray, shape (2,)
            Coordinates of the second endpoint of segment 1.
        q1 : np.ndarray, shape (2,)
            Coordinates of the first endpoint of segment 2.
        q2 : np.ndarray, shape (2,)
            Coordinates of the second endpoint of segment 2.

        Returns
        -------
        float
            The minimum Euclidean distance between the two segments.
        """
        u, v, w = p2 - p1, q2 - q1, p1 - q1
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

        # Loop over each particle
        for i in range(self.N_particles):
            C_sum = 0.0
            # Loop over each particle
            for j in range(self.N_particles):
                if i!=j:
                    # NP - NP - Capacitance
                    val     =   self.mutal_capacitance_adjacent_spheres_sinh(eps_r, self.radius_vals[i], self.radius_vals[j], self.dist_matrix[i,j])
                    C_sum   +=  val
                    self.capacitance_matrix[i,j] = -val

            for j in range(self.N_electrodes):
                # NP - Electrode - Capacitance
                if j not in self.floating_indices:
                    val     =   self.mutal_capacitance_adjacent_spheres_sinh(eps_r, self.radius_vals[i], self.ELECTRODE_RADIUS, self.electrode_dist_matrix[j,i])
                    C_sum   +=  val
                    
            # Self Capacitance
            C_sum += self.self_capacitance_sphere(eps_s, self.radius_vals[i])
            
            # Set diagonal element (total capacitance)
            self.capacitance_matrix[i,i] = C_sum
        
        try:
            # Calculate the inverse capacitance matrix
            self.inv_capacitance_matrix = np.linalg.inv(self.capacitance_matrix)
        except np.linalg.LinAlgError as e:
            raise RuntimeError("Failed to invert capacitance matrix. Matrix may be singular.") from e
        
    def calc_electrode_capacitance_matrix(self):

        # Boolean mask: True for constant electrodes, False for floating
        constant_mask = np.ones(self.N_electrodes, dtype=bool)
        constant_mask[self.floating_indices] = False

        # C_lead[j, i] = C between electrode j and particle i (zero if floating)
        C_lead = np.zeros((self.N_electrodes, self.N_particles))
        for j in np.nonzero(constant_mask)[0]:
            for i in range(self.N_particles):
                C_lead[j, i] = self.mutal_capacitance_adjacent_spheres_sinh(self.eps_r,self.radius_vals[i],
                                                                            self.ELECTRODE_RADIUS,self.electrode_dist_matrix[j, i])

        # C_self[i] = self‑capacitance of sphere i
        C_self = np.array([self.self_capacitance_sphere(self.eps_s, r) for r in self.radius_vals])
        self.self_capacitance               = C_self        # shape (N_particles,)
        self.electrode_capacitance_matrix   = C_lead        # shape (N_electrodes, N_particles)
        
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
        
        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        # C_lead.T @ V_e is shape (N_particles,)
        self.charge_vector = self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g

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

        V_e = voltage_values[:self.N_electrodes]
        V_g = voltage_values[-1]

        return self.electrode_capacitance_matrix.T.dot(V_e) + self.self_capacitance * V_g
        
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
    
    
    def return_capacitance_matrix(self) -> np.ndarray:
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
    
    def return_electrode_capacitance_matrix(self) -> np.ndarray:
        """Get the network electrode capacitance matrix.

        Returns
        -------
        np.array
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
    
    def return_self_capacitance(self) -> np.ndarray:
        """Get the self capacitance of each NP.

        Returns
        -------
        np.array
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
