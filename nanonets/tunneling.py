import numpy as np
from . import electrostatic
from typing import Tuple, List, Optional

class NanoparticleTunneling(electrostatic.NanoparticleElectrostatic):
    """
    Simulates single-electron tunneling dynamics in nanoparticle networks.

    This class extends NanoparticleElectrostatic to add functionality for:
    - Tunneling events (NP-NP, NP-electrode)
    - Resistive junctions (static or dynamic)
    - Temperature effects on tunneling
    - Efficient bookkeeping of event indices for KMC and rate equation methods

    Indexing Scheme
    ---------------
    - Nanoparticles are indexed from `self.N_electrodes` to `self.N_electrodes + self.N_particles - 1`
    - Electrodes are indexed from `0` to `self.N_electrodes - 1`
    - Tunneling events are indexed as (i → j) using arrays `adv_index_rows`, `adv_index_cols`
    - All nodes (NPs and electrodes) can be mapped to a dense index for Laplacian/conductance calculations

    Physical Constants (class attributes)
    -------------------------------------
    ELE_CHARGE_A_C : float
        Elementary charge [attoCoulombs, aC = 1e-18 C]
    ELE_CHARGE_C : float
        Elementary charge [C]
    KB_AJ_PER_K : float
        Boltzmann constant [aJ/K = 1e-18 J/K]
    KB_EV_PER_K : float
        Boltzmann constant [eV/K]
    MIN_RESISTANCE_MOHM : float
        Minimum allowed tunnel resistance [MΩ]

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles in the network
    N_electrodes : int
        Number of electrodes
    N_junctions : int
        Number of nearest-neighbor junctions per nanoparticle (max degree)
    net_topology : np.ndarray
        Network topology matrix (from parent class)
    adv_index_rows : np.ndarray
        Origin node indices for all possible tunnel events (i→j)
    adv_index_cols : np.ndarray
        Target node indices for all possible tunnel events (i→j)
    potential_vector : np.ndarray
        Current network potential for all nodes [V]
    const_capacitance_values : np.ndarray
        Precomputed free energy terms for tunneling events [(aC)^2/aF]
    resistances : np.ndarray
        1D array of tunnel resistances [MΩ], one per tunneling event (always undirected)
    conductance_matrix : np.ndarray
        Network conductance (Laplacian) matrix [1/MΩ], including all NPs and electrodes

    Methods
    -------
    init_adv_indices()
        Build index arrays for all valid tunneling events (NP-NP, NP-electrode)
    init_potential_vector(voltage_values)
        Set up network potential using electrode/gate voltages
    init_const_capacitance_values()
        Precompute capacitance terms for all tunneling events (for free energy)
    init_junction_resistances(R, Rstd)
        Initialize random tunnel resistances (truncated normal, undirected)
    update_junction_resistances(junctions, R)
        Update resistances for specific (undirected) junctions
    update_junction_resistances_at_random(N, R)
        Update resistances for N random (undirected) junctions
    update_nanoparticle_resistances(nanoparticles, R)
        Update all resistances to/from a given set of nanoparticles
    build_conductance_matrix()
        Construct the full network conductance (Laplacian) matrix
    get_potential_vector()
        Return the current potential vector [V]
    get_const_capacitance_values()
        Return capacitance terms for tunneling free energy [(aC)^2/aF]
    get_particle_electrode_count()
        Return (N_electrodes, N_particles)
    get_advanced_indices()
        Return tunneling event indices (origin, target)
    get_const_temperatures(T)
        Return temperature-dependent array [aJ] for each tunneling event
    get_tunneling_rate_prefactor()
        Return the rate prefactor array: resistance × e^2 [(MΩ)·(aC)^2] for each event
    get_conductance_matrix()
        Return the current conductance (Laplacian) matrix [1/MΩ]

    Notes
    -----
    - All units are SI (capacitance in aF, charge in aC, resistance in MΩ, temperature in aJ/K).
    - All tunnel resistances are always treated as undirected (R_ij = R_ji).
    - Index conventions are consistent for fast, vectorized simulation.
    - Intended for use in kinetic Monte Carlo, stochastic, or master equation modeling of nanoparticle networks.
    """
    
    # Physical constants
    ELE_CHARGE_A_C      = 0.160217662       # [aC] (attoCoulombs, 1e-18 C)
    ELE_CHARGE_C        = 1.60217662e-19    # [C]   (Coulombs)
    KB_AJ_PER_K         = 1.380649e-5       # [aJ/K] (attoJoules per Kelvin, 1e-18 J/K)
    KB_EV_PER_K         = 8.617333262e-5    # [eV/K] (electronvolts per Kelvin) 
    MIN_RESISTANCE_MOHM = 1.0               # [MOhm]

    def __init__(self, electrode_type: Optional[List[str]] = None, seed: Optional[int] = None) -> None:
        """Initialize tunneling class.

        Parameters
        ----------
        electrode_type : List[str], optional
            List specifying electrode types ('constant' or 'floating')
        seed : int, optional
            Random seed for reproducibility, by default None
        """
        super().__init__(electrode_type, seed)
        
    def init_adv_indices(self) -> None:
        """
        Initialize advanced indices for all possible tunneling events (i→j).

        This sets up two arrays:
        - self.adv_index_rows: origin indices (i) for each possible tunneling event (i→j)
        - self.adv_index_cols: destination indices (j) for each event (i→j)

        These indices can be used for vectorized updates of tunneling-related arrays (e.g., rates, resistances).

        Notes
        -----
        - Only valid, connected pairs are included.
        - Nanoparticle nodes are indexed from self.N_electrodes to self.N_electrodes+self.N_particles-1.
        - Electrodes are indexed from 0 to self.N_electrodes-1 (shifted for array construction).
        - Updates self.adv_index_rows and self.adv_index_cols as 1D arrays.
        
        Raises
        ------
        RuntimeError
            If the network topology or basic attributes are not initialized.
        """

        # Ensure basic attributes
        if not hasattr(self, "net_topology") or not hasattr(self, "N_particles") or not hasattr(self, "N_electrodes") or not hasattr(self, "N_junctions"):
            raise RuntimeError("Network not initialized. Run network setup first.")
        
        # Prepare a unified connections array: [electrodes | nanoparticles], shape: (N_electrodes+N_particles, N_junctions+1)
        connections = np.full((self.N_particles + self.N_electrodes, self.N_junctions + 1), self.NO_CONNECTION, dtype=int)
        # Place nanoparticle connectivity (topology) in lower block
        connections[self.N_electrodes:, :] = self.net_topology

        # Map which nanoparticle is attached to each electrode (first column of net_topology)
        nth_e, nth_np = 1, 0
        while (nth_np < self.N_particles) and (nth_e <= self.N_electrodes):
            if int(self.net_topology[nth_np, 0]) == nth_e:
                # The nth_e electrode is connected to nth_np nanoparticle
                connections[nth_e - 1, 1] = nth_np
                nth_e += 1
                nth_np = 0
                continue
            nth_np += 1

        # Offset indices for compact event indexing:
        # - Electrodes: [0 ... N_electrodes-1]
        # - Nanoparticles: [N_electrodes ... N_electrodes+N_particles-1]
        connections[:, 0]  -= 1  # if electrode number is 1-based, shift to 0-based
        connections[:, 1:] += self.N_electrodes

        # Build tunneling event lists: for each node, find all valid partners
        adv_index_cols = [list(row[row >= 0].astype(int)) for row in connections]
        adv_index_rows = [len(col_list) * [i] for i, col_list in enumerate(adv_index_cols)]

        # Flatten lists for fast vectorized lookup (event i→j: self.adv_index_rows[k], self.adv_index_cols[k])
        self.adv_index_cols = np.array([item for sublist in adv_index_cols for item in sublist], dtype=int)
        self.adv_index_rows = np.array([item for sublist in adv_index_rows for item in sublist], dtype=int)
                
    def init_potential_vector(self, voltage_values: np.ndarray) -> None:
        """
        Initialize the full potential vector for the network, setting electrode and gate voltages.

        The potential vector is organized as:
        - First N_electrodes entries: electrode voltages (from voltage_values)
        - Remaining N_particles entries: initialized to zero (to be filled in by electrostatic solver)

        Parameters
        ----------
        voltage_values : np.ndarray or list
            Array of shape (N_electrodes + 1,) = [V_e1, V_e2, ..., V_eN, V_G]
            where V_G is the gate voltage.

        Raises
        ------
        ValueError
            If voltage_values length does not match number of electrodes + 1.
        RuntimeError
            If N_particles or N_electrodes are not set.
        """
        if not hasattr(self, "N_particles") or not hasattr(self, "N_electrodes"):
            raise RuntimeError("Network not initialized: N_particles and N_electrodes must be set.")

        voltage_values = np.asarray(voltage_values)
        if voltage_values.shape[0] != self.N_electrodes + 1:
            raise ValueError(f"Expected {self.N_electrodes + 1} voltage values, got {voltage_values.shape[0]}.")
        
        self.potential_vector = np.zeros(self.N_electrodes + self.N_particles)
        self.potential_vector[0:self.N_electrodes] = voltage_values[:-1]

    def init_const_capacitance_values(self) -> None:
        """
        Precompute constant capacitance terms for tunneling free energy calculations.

        This computes the (C_ii + C_jj - 2*C_ij) * (e^2 / 2) for all possible tunneling
        events, using advanced index arrays. Electrode rows/columns are masked out.

        Requires:
        ---------
        - self.adv_index_rows, self.adv_index_cols : index arrays for all tunneling events
        - self.inv_capacitance_matrix : inverse capacitance matrix (NxN)
        - self.ele_charge : elementary charge [aC]

        Stores:
        -------
        - self.const_capacitance_values : ndarray, precomputed constant terms for all tunnel events

        Raises
        ------
        RuntimeError
            If necessary attributes are missing.
        """
        if not (hasattr(self, "adv_index_rows") and hasattr(self, "adv_index_cols") and hasattr(self, "inv_capacitance_matrix")):
            raise RuntimeError("Must call init_adv_indices and calc_capacitance_matrix first.")
        
        # Indices relative to nanoparticle-only part (exclude electrodes)
        row_i = self.adv_index_rows - self.N_electrodes
        col_i = self.adv_index_cols - self.N_electrodes
        
        # Only use valid nanoparticle indices (exclude electrodes)
        row_mask = (row_i >= 0).astype(int)
        col_mask = (col_i >= 0).astype(int)

        # Precompute capacitance terms
        cap_ii = self.inv_capacitance_matrix[row_i, row_i] * row_mask * self.ELE_CHARGE_A_C**2 / 2
        cap_jj = self.inv_capacitance_matrix[col_i, col_i] * col_mask * self.ELE_CHARGE_A_C**2 / 2
        cap_ij = self.inv_capacitance_matrix[row_i, col_i] * row_mask * col_mask * self.ELE_CHARGE_A_C**2 / 2

        self.const_capacitance_values = (cap_ii + cap_jj - 2 * cap_ij)

    def init_junction_resistances(self, R: float = 25, Rstd: float = 0) -> None:
        """
        Initialize a 1D array of tunnel resistances (in MΩ) for each unique NP-NP or NP-electrode junction.
        Ensures all resistances are undirected and above a minimum value.
        
        Parameters
        ----------
        R : float
            Mean tunnel resistance [MΩ]
        Rstd : float
            Standard deviation of resistances [MΩ]
        """
        n_junctions = len(self.adv_index_rows)
        if R <= 0:
            raise ValueError(f"Mean resistance must be positive, got {R}")
        if Rstd < 0:
            raise ValueError(f"Std deviation must be non-negative, got {Rstd}")

        # Sample resistances from normal distribution, resample if below minimum
        resistance = self.rng.normal(R, Rstd, size=n_junctions)
        while np.any(resistance < self.MIN_RESISTANCE_MOHM):
            bad = resistance < self.MIN_RESISTANCE_MOHM
            resistance[bad] = self.rng.normal(R, Rstd, size=np.sum(bad))

        # Enforce undirected (symmetry) property
        self.resistances = self._ensure_undirected_resistances(resistance, average=True)

    def _ensure_undirected_resistances(self, resistances: np.ndarray, average: bool = False) -> np.ndarray:
        """
        Ensures that resistances are undirected by making R(i, j) = R(j, i).
        
        Parameters
        ----------
        resistances : np.ndarray
            1D array of resistance values corresponding to the junction pairs.
        average : bool, optional
            If True, the resistance values for both directions (i, j) and (j, i) are averaged.
            If False, the resistance value of (i, j) overwrites (j, i). Default is False.

        Returns
        -------
        np.ndarray
            Updated resistance values with undirected property enforced.
        """
        # Create a dictionary to map (i, j) to resistance
        pair_to_index = {}
        for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols)):
            pair_to_index[(i, j)] = idx
        
        # Iterate through each resistance and enforce symmetry
        for idx, (i, j) in enumerate(zip(self.adv_index_rows, self.adv_index_cols)):
            # Get the reverse pair (j, i)
            reverse_pair = (j, i)
            if reverse_pair in pair_to_index:
                reverse_idx = pair_to_index[reverse_pair]
                # Ensure symmetry
                if average:
                    average_resistance          = (resistances[idx] + resistances[reverse_idx]) / 2.0
                    resistances[idx]            = average_resistance
                    resistances[reverse_idx]    = average_resistance
                else:
                    resistances[reverse_idx]    = resistances[idx]

        return resistances
    
    def update_junction_resistances(self, junctions: List[Tuple[int, int]], R: float = 25) -> None:
        """
        Update tunnel resistances for specific junctions.

        Parameters
        ----------
        junctions : List[Tuple[int, int]]
            List of (origin, target) NP index pairs to update (indices relative to NP array).
        R : float, optional
            New resistance value [MΩ], by default 25

        Raises
        ------
        ValueError
            If a specified junction does not exist.
        """

        for junc in junctions:
            # Update forward direction
            a   = np.where(self.adv_index_rows == junc[0] + self.N_electrodes)[0]
            b   = np.where(self.adv_index_cols == junc[1] + self.N_electrodes)[0]
            idx = np.intersect1d(a,b)[0]
            self.resistances[idx] = R

            # Update reverse direction
            a   = np.where(self.adv_index_cols == junc[0] + self.N_electrodes)[0]
            b   = np.where(self.adv_index_rows == junc[1] + self.N_electrodes)[0]
            idx = np.intersect1d(a,b)[0]
            self.resistances[idx] = R

    def update_junction_resistances_at_random(self, N: int, R: float = 25.0, Rstd: float = 0.0) -> None:
        """
        Randomly select N unique undirected NP-NP junctions and update their tunnel resistances.

        Parameters
        ----------
        N : int
            Number of distinct (undirected) junctions to modify.
        R : float, optional
            Mean resistance value [MΩ] (default: 25).
        Rstd : float, optional
            Standard deviation for resistance values [MΩ] (default: 0).

        Raises
        ------
        ValueError
            If N is greater than the total number of available undirected junctions.
        """

        all_pairs   = []
        seen        = set()
        for row, col in zip(self.adv_index_rows, self.adv_index_cols):
            i = row - self.N_electrodes
            j = col - self.N_electrodes
            if ((i < j) and (i >= 0) and (j >= 0)):
                pair = (i, j)
                if pair not in seen:
                    all_pairs.append((i,j))
                    seen.add(pair)

        if N > len(all_pairs):
            raise ValueError(f"Requested {N} junctions, but only {len(all_pairs)} available.")
        
        # Randomly choose N distinct junctions
        chosen_pairs = self.rng.choice(a=all_pairs, size=N, replace=False)
        chosen_pairs = [tuple(pair) for pair in chosen_pairs]

        # Generate resistance values
        if Rstd > 0:
            new_resistances = self.rng.normal(R, Rstd, size=N)
            # Resample if any below minimum
            while np.any(new_resistances < self.MIN_RESISTANCE_MOHM):
                bad = new_resistances < self.MIN_RESISTANCE_MOHM
                new_resistances[bad] = self.rng.normal(R, Rstd, size=np.sum(bad))
        else:
            new_resistances = np.full(N, R)
        
        # Set each chosen junction (and its reverse) to the sampled value
        for (pair, new_R) in zip(chosen_pairs, new_resistances):
            self.update_junction_resistances([pair], new_R)

    # TODO: Allow Gauss distributed
    def update_nanoparticle_resistances(self, nanoparticles: List[int], R: float = 25) -> None:
        """
        Set all resistances in self.resistances corresponding to jumps
        *to or from* the specified nanoparticles to a new value.

        Parameters
        ----------
        nanoparticles : List[int]
            Indices of nanoparticles for which all associated resistances should be updated.
        R : float, optional
            New resistance value [MΩ] to assign, by default 25.

        Raises
        ------
        RuntimeError
            If resistances have not been initialized.
        ValueError
            If any nanoparticle index is out of bounds.
        """
        if not hasattr(self, "resistances"):
            raise RuntimeError("Resistances have not been initialized. Call init_resistances first.")

        for idx in nanoparticles:
            adv_idx = idx + self.N_electrodes
            # Set for all jumps to and from this NP
            self.resistances[np.where(self.adv_index_cols == adv_idx)[0]] = R
            self.resistances[np.where(self.adv_index_rows == adv_idx)[0]] = R
    
    def build_conductance_matrix(self) -> None:
        """
        Build and return the conductance (Laplacian) matrix for the network [S]

        Each off-diagonal entry -g_{ij} represents a conductance between nodes i and j.
        Diagonal entries sum all conductances connected to node i.
        The resulting matrix can be used for network current/voltage calculations.

        Raises
        ------
        RuntimeError
            If resistances are not initialized.

        Notes
        -----
        - The node ordering is: [nanoparticles..., electrodes...]
        - Node indices correspond to their new positions in the matrix via raw2dense.
        """
        if not hasattr(self, "resistances"):
            raise RuntimeError("Resistances have not been initialized.")

        src = self.adv_index_rows.copy()
        tgt = self.adv_index_cols.copy()

        # Get sorted list of all node indices (NPs >= 0, electrodes < 0)
        np_nodes    = sorted({idx for idx in np.concatenate([src, tgt]) if idx >= 0})
        el_nodes    = sorted({idx for idx in np.concatenate([src, tgt]) if idx <  0})
        total_n     = len(np_nodes) + len(el_nodes)

        # Map raw node indices to their position in the matrix
        raw2dense   = {raw: i for i, raw in enumerate(np_nodes + el_nodes)}
        cond_matrix = np.zeros((total_n, total_n))

        for s_raw, t_raw, R_l in zip(src, tgt, self.resistances):
            g = 1.0 / R_l
            i = raw2dense[s_raw]
            j = raw2dense[t_raw]

            # symmetric update
            cond_matrix[i, i] += g
            cond_matrix[j, j] += g
            cond_matrix[i, j] -= g
            cond_matrix[j, i] -= g

        self.conductance_matrix = cond_matrix*1e-6

    def init_transfer_coeffs(self, output_electrode: int = None) -> None:
        """
        Calculate and store the current transfer coefficients for the network.

        This computes, for each electrode, the fraction of injected current
        that reaches a selected output electrode, using the conductance matrix.

        Parameters
        ----------
        output_electrode : int, optional
            Index of the output electrode (0-based). If None, uses the last electrode.

        Raises
        ------
        RuntimeError
            If conductance matrix is not initialized.
        ValueError
            If output electrode index is invalid.

        Stores
        ------
        self.transfer_coeffs : np.ndarray
            1D array, shape (N_electrodes,), transfer coefficients (unitless).
            self.transfer_coeffs[i] gives the proportion of current at the output
            electrode when 1V is applied at electrode i.
        """
        if not hasattr(self, "conductance_matrix"):
            raise RuntimeError("Conductance matrix not calculated. Call build_conductance_matrix first.")

        N_e     = self.N_electrodes
        N_p     = self.N_particles
        n_nodes = N_e + N_p

        # Output electrode: by default, the last one
        if output_electrode is None:
            out_idx = N_e - 1
        else:
            out_idx = int(output_electrode)
            if not (0 <= out_idx < N_e):
                raise ValueError(f"Invalid output_electrode index: {out_idx} (must be in 0..{N_e-1})")

        # Indices for unknowns (NPs) and knowns (electrodes)
        u_idx   = np.arange(self.N_electrodes, n_nodes)
        k_idx   = np.arange(self.N_electrodes)

        Y       = self.conductance_matrix
        Y_uu    = Y[np.ix_(u_idx, u_idx)]
        Y_uk    = Y[np.ix_(u_idx, k_idx)]

        # Precompute A = -Y_uu^{-1} Y_uk
        inv_Yuu = np.linalg.inv(Y_uu)
        A       = -inv_Yuu @ Y_uk  # (N_p, N_e)

        # Output electrode row (current into output for 1V at each electrode)
        Y_out_k     = Y[out_idx, k_idx]
        Y_out_u     = Y[out_idx, u_idx]
        indirect    = Y_out_u @ A

        transfer_coeffs = -Y_out_k - indirect
        transfer_coeffs[out_idx] = 0.0  # By convention: self-coupling is zero

        self.transfer_coeffs = transfer_coeffs

    def calibrate_electrodes(self, ref_electrodes: List[int], ref_current: float = 1e-9,
                             alpha: float = 0.3, use_mean: bool = False):
        """
        Calibrate voltage amplitudes for all electrodes to achieve specified output current scaling.

        Reference electrodes are set to achieve `ref_current` at the output (summed).
        The remaining electrodes ("controls") are set so their current at the output
        is a fraction (`alpha`) of the reference output current.

        Parameters
        ----------
        ref_electrodes : List[int]
            List of indices (0-based) for reference/input electrodes.
        ref_current : float, optional
            Desired reference output current [A] at the output (default: 1e-9).
        alpha : float, optional
            Ratio of control output current to reference output current (default: 0.3).
        use_mean : bool, optional
            If True, each reference electrode is calibrated to drive ref_current
            individually (voltages will be higher). If False, reference voltages
            are set so their combined effect gives ref_current (default).

        Stores
        ------
        self.delta_V : np.ndarray
            1D array of length N_electrodes, voltage amplitudes [V] for each electrode.

        Raises
        ------
        RuntimeError
            If transfer coefficients have not been initialized.
        ValueError
            If no valid reference electrodes are provided.
        """

        if not hasattr(self, "transfer_coeffs"):
            raise RuntimeError("Transfer coefficients not initialized. Call init_transfer_coeffs first.")
        
        delta_V = np.zeros_like(self.transfer_coeffs)
        all_electrodes = set(range(self.N_electrodes))
        control_electrodes = [i for i in all_electrodes if i not in ref_electrodes]

        # Reference voltage calculation
        if use_mean:
            G_input = np.mean(self.transfer_coeffs[ref_electrodes])
            if G_input == 0:
                raise ValueError("Mean transfer coefficient for reference electrodes is zero.")
            V_on = ref_current / G_input
        else:
            G_input = np.sum(self.transfer_coeffs[ref_electrodes])
            if G_input == 0:
                raise ValueError("Sum of transfer coefficients for reference electrodes is zero.")
            V_on = ref_current / G_input

        for k in ref_electrodes:
            delta_V[k] = V_on

        # Control (non-reference) electrode calibration
        for k in control_electrodes:
            coeff = self.transfer_coeffs[k]
            if coeff != 0:
                delta_V[k] = alpha * abs(ref_current) / abs(coeff)
            else:
                delta_V[k] = 0.0

        self.delta_V = delta_V

    def get_potential_vector(self) -> np.ndarray:
        """
        Returns
        -------
        potential_vector : ndarray
            Potential values for electrodes and nanoparticles [V]
        """

        return self.potential_vector

    def get_const_capacitance_values(self) -> np.ndarray:
        """
        Returns
        -------
        const_capacitance_values : ndarray
            Capacitance terms for tunneling free energy calculation [aC^2/aF]
        """
        return self.const_capacitance_values

    def get_particle_electrode_count(self) -> Tuple[int, int]:
        """
        Returns
        -------
        N_particles : int
            Number of nanoparticles
        N_electrodes : int
            Number of electrodes
        """

        return self.N_particles, self.N_electrodes
    
    def get_advanced_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        adv_index_rows : ndarray of shape (n_tunnel_events,)
            Origin nanoparticles (i) in tunneling events i→j
        adv_index_cols : ndarray of shape (n_tunnel_events,)
            Target nanoparticles (j) in tunneling events i→j
        """
        return self.adv_index_rows, self.adv_index_cols
    
    def get_const_temperatures(self, T: float = 0.28) -> np.ndarray:
        """
        Parameters
        ----------
        T : float
            Network temperature [K]

        Returns
        -------
        T_arr : ndarray
            Array of thermal energies for each tunneling event [aJ]
            (T * kB, one per tunneling event)
        """
        return np.repeat(T * self.KB_AJ_PER_K, len(self.adv_index_rows))
    
    def get_tunneling_rate_prefactor(self) -> np.ndarray:
        """
        Returns
        -------
        ndarray
            1D array of rate prefactors (R_{ij} * e^2) [(MΩ·(aC)^2)]
            for each tunneling event (i→j).
        """
        if not hasattr(self, 'resistances'):
            raise RuntimeError("Junction resistances not initialized. Call init_junction_resistances first.")
        return self.resistances * self.ELE_CHARGE_A_C * self.ELE_CHARGE_A_C * 1e-12  
    
    def get_resistance(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            1D array of resistance values for each tunneling event (i->j)
        """
        if not hasattr(self, 'resistances'):
            raise RuntimeError("Junction resistances not initialized. Call init_junction_resistances first.")
        return self.resistances 
    
    def get_conductance_matrix(self) -> np.ndarray:
        """
        Get the network conductance (Laplacian) matrix.

        Returns
        -------
        np.ndarray
            Symmetric conductance matrix [1/MΩ], shape: (N_total, N_total),
            where N_total = N_particles + N_electrodes

        Raises
        ------
        RuntimeError
            If conductance matrix hasn't been calculated (call build_conductance_matrix first).
        """
        if not hasattr(self, 'conductance_matrix'):
            raise RuntimeError("Conductance matrix not calculated. Call build_conductance_matrix first.")
        return self.conductance_matrix
    
    def get_transfer_coeffs(self) -> np.ndarray:
        """
        Get the vector of transfer coefficients for the current output electrode.

        Returns
        -------
        np.ndarray
            Transfer coefficients, shape (N_electrodes,), unitless
        Raises
        ------
        RuntimeError
            If transfer_coeffs are not initialized (call init_transfer_coeffs first).
        """
        if not hasattr(self, "transfer_coeffs"):
            raise RuntimeError("Transfer coefficients not initialized. Call init_transfer_coeffs first.")
        return self.transfer_coeffs

    def get_delta_V(self) -> np.ndarray:
        """
        Get the calibrated electrode voltage ranges.

        Returns
        -------
        np.ndarray
            Calibrated voltages for each electrode (same order as self.transfer_coeffs).
        """
        if not hasattr(self, 'delta_V'):
            raise RuntimeError("Electrode voltages not calibrated. Call calibrate_electrodes first.")
        return self.delta_V