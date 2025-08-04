import tunneling
import monte_carlo
import numpy as np
import pandas as pd
import os.path

class Simulation(tunneling.NanoparticleTunneling):
    """
    Simulation class for nanoparticle networks with single-electron effects.

    This class builds on NanoparticleTunneling to construct a full physical
    nanoparticle device model, including:
        - Topology (lattice or random graph)
        - Electrostatics (capacitance matrix, particle radii, packing)
        - Tunnel junction properties (resistances)
        - Electrode placement and types (constant/floating)
        - Support for device "heterogeneity" via two NP types or resistances

    Attributes
    ----------
    network_topology : str
        Type of network, either 'lattice' or 'random'.
    res_info : dict
        Resistance parameters for primary NP type.
    res_info2 : dict or None
        Resistance parameters for second NP type (if present).
    folder : str
        Folder where simulation outputs are stored.
    path1, path2, path3 : str
        Full file paths for outputting results.
    [All attributes of NanoparticleTunneling are also present.]

    Parameters
    ----------
    topology_parameter : dict
        Dictionary specifying topology, number of particles, electrode positions, and electrode types.
        Required keys depend on topology:
            Lattice: 'Nx', 'Ny', 'e_pos', 'electrode_type'
            Random:  'Np', 'Nj', 'e_pos', 'electrode_type'
    folder : str, optional
        Path to output folder for saving results.
    add_to_path : str, optional
        Additional string appended to all output file names.
    res_info : dict, optional
        Resistance parameters for primary NP type. Example:
            {"mean_R": 25.0, "std_R": 0.0, "dynamic": False}
    res_info2 : dict, optional
        Resistance parameters for second NP type.
    np_info : dict, optional
        Physical parameters for first NP type. Example:
            {"eps_r": 2.6, "eps_s": 3.9, "mean_radius": 10.0, "std_radius": 0.0, "np_distance": 1.0}
    np_info2 : dict, optional
        Physical parameters for second NP type (may include 'np_index' for affected particles).
    seed : int, optional
        Seed for random number generation.
    high_C_output : bool, optional
        If True, adds an additional high-capacitance output electrode.
    kwargs : dict, optional
        Extra options (such as del_n_junctions: int for sparse/disordered networks).

    Raises
    ------
    ValueError
        If an unsupported network topology is requested.

    Notes
    -----
    - Lattice topologies are square 2D grids; random topologies use Delaunay triangulation.
    - Supports two nanoparticle types for heterogeneous materials/devices.
    - Sets up all matrices and data needed for electrostatics and dynamics.
    - See parent class docstrings for further details on physics and data.
    """

    def __init__(self, topology_parameter : dict, folder: str = '', add_to_path: str = "", res_info: dict = None, res_info2: dict = None,
                 np_info: dict = None, np_info2: dict = None, seed: int = None, high_C_output: bool = False, **kwargs):
        """
        Defines network topology, electrostatic properties, and tunneling junctions for a given topology.

        Parameters
        ----------
        topology_parameter : dict
            Dictionary including number of nanoparticles, electrode positions, and electrode types.
            For lattice: must include 'Nx', 'Ny', 'e_pos', 'electrode_type'.
            For random: must include 'Np', 'e_pos', 'electrode_type'.
        folder : str, optional
            Directory where simulation results are saved (default: '').
        add_to_path : str, optional
            String appended to output file names (default: "").
        res_info : dict, optional
            Resistance parameters for primary NP type. (default: mean_R=25.0, std_R=0.0, dynamic=False)
        res_info2 : dict, optional
            Resistance parameters for secondary NP type (if any).
        np_info : dict, optional
            Parameters for first nanoparticle type (default: see code).
        np_info2 : dict, optional
            Parameters for second nanoparticle type (may include 'np_index' for which NPs are affected).
        seed : int, optional
            Random seed for reproducibility.
        high_C_output : bool, optional
            Whether to add a high-capacitance output electrode (default: False).
        kwargs : dict
            Additional keyword arguments (e.g., 'del_n_junctions').

        Raises
        ------
        ValueError
            If required keys are missing, or unsupported topology is requested.

        Notes
        -----
        - All attributes for electrostatics, topology, and tunneling are set up automatically.
        - Network can have two nanoparticle types for heterogeneity.
        - All physical quantities are initialized, and key file paths are pre-set.
        """

        # --- Electrode type and inheritance ---
        electrode_type  = topology_parameter['electrode_type']
        super().__init__(electrode_type, seed)

        # --- Topology type ---
        if 'Nx' in topology_parameter:
            self.network_topology = 'lattice'
        else:
            self.network_topology = 'random'

        # --- Default NP info ---
        if np_info is None:
            np_info = {
                "eps_r"         : 2.6,  # Permittivity of molecular junction 
                "eps_s"         : 3.9,  # Permittivity of oxide layer
                "mean_radius"   : 10.0, # average nanoparticle radius
                "std_radius"    : 0.0   # standard deviation of nanoparticle radius
            }

        # --- Default Resistance info ---
        if res_info is None:
            res_info = {
                "mean_R"    : 25.0, # Average resistance
                "std_R"     : 0.0,  # Standard deviation of resistances
                "dynamic"   : False # Dynamic or constant resistances
            }
        
        # --- Dynamic Junction Resistances ---
        self.dynamic_resistances = res_info['dynamic']
        if self.dynamic_resistances:
            self.dynamic_resistances_info = res_info

        # --- Lattice topology ---
        if self.network_topology == "lattice":
            # Path variable
            path_var = f'Nx={topology_parameter["Nx"]}_Ny={topology_parameter["Ny"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # --- Topology ---
            self.lattice_network(N_x=topology_parameter["Nx"], N_y=topology_parameter["Ny"])
            self.add_electrodes_to_lattice_net(topology_parameter["e_pos"])
                                    
        # --- Random Topology ---
        elif self.network_topology == "random":
            # Path variable
            path_var = f'Np={topology_parameter["Np"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # --- Topology ---
            self.random_network(N_particles=topology_parameter["Np"])
            self.add_electrodes_to_random_net(electrode_positions=topology_parameter["e_pos"])
        
        else:
            raise ValueError("Only 'lattice' and 'random' topologies are supported.")
        
        if high_C_output:
            self.add_np_to_output()

        # --- Electrostatics ---
        self.init_nanoparticle_radius(np_info['mean_radius'], np_info['std_radius'])
        if np_info2 is not None:
            self.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'])
        self.pack_planar_circles()
        self.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'])
        self.calc_electrode_capacitance_matrix()
        self.init_const_capacitance_values()

        # --- Tunneling ---
        self.init_adv_indices()
        self.init_junction_resistances(res_info['mean_R'], res_info['std_R'])
        if res_info2 is not None:
            self.update_junction_resistances_at_random(res_info2['N'], res_info2['mean_R'], res_info2['std_R'])

        # --- Path ---
        self.folder = folder
        self.path1  = folder + path_var
        self.path2  = folder + 'mean_state_'    + path_var
        self.path3  = folder + 'net_currents_'  + path_var

    def run_const_voltages(self, voltages : np.ndarray, target_electrode : int, T_val: float = 0.0, sim_dic: dict = None, save_th: int = 10, verbose: bool = False):
        """
        Run a kinetic Monte Carlo simulation at fixed electrode voltages, estimating either the steady-state
        current or, for floating electrodes, the equilibrium potential at a selected electrode.

        Parameters
        ----------
        voltages : np.ndarray
            2D array of electrode voltages. Each row is a distinct set of electrode voltages [V].
        target_electrode : int
            Index of the electrode for which to record current (constant) or potential (floating).
        T_val : float, optional
            Network temperature [K]. Default: 0.0.
        sim_dic : dict, optional
            Dictionary of simulation parameters:
                error_th        : Target relative error of the main observable.
                max_jumps       : Maximum KMC steps per voltage point.
                eq_steps        : Number of equilibration KMC steps before measurement.
                jumps_per_batch : KMC steps per batch.
                kmc_counting    : If True, use event-counting for current.
                min_batches     : Minimum measurement batches.
        save_th : int, optional
            Frequency (in voltage steps) with which simulation data is saved to file.
        verbose : bool, optional
            If True, records and stores extra simulation data (longer runtime, larger outputs).

        Returns
        -------
        None
            Results are saved to file and/or stored in class attributes.
        """

        # Round voltages to 0.01 mV
        voltages = np.round(voltages,5)
        
        # --- Default Simulation Parameter ---
        if sim_dic is None:
            sim_dic =   {
                "error_th"        : 0.05,
                "max_jumps"       : 10000000,
                "eq_steps"        : 100000,
                "jumps_per_batch" : 5000,
                "kmc_counting"    : False,
                "min_batches"     : 5
            }

        # --- Get Simulation Parameter ---
        error_th        = sim_dic['error_th']
        max_jumps       = sim_dic['max_jumps']
        eq_steps        = sim_dic['eq_steps']
        jumps_per_batch = sim_dic['jumps_per_batch']
        kmc_counting    = sim_dic['kmc_counting']
        min_batches     = sim_dic['min_batches']

        # Identify floating electrodes and detect if target electrode is floating
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        output_potential    = (self.electrode_type[target_electrode] == 'floating')
        
        # Init storage lists
        self.clear_simulation_outputs()
        
        j = 0
        for i, voltage_values in enumerate(voltages):
            
            # --- Update network electrostatics ---
            self.init_charge_vector(voltage_values=voltage_values)
            self.init_potential_vector(voltage_values=voltage_values)

            # --- Get all KMC model input arrays ---
            inv_capacitance_matrix          = self.get_inv_capacitance_matrix()
            charge_vector                   = self.get_charge_vector()
            potential_vector                = self.get_potential_vector()
            const_capacitance_values        = self.get_const_capacitance_values()
            N_particles, N_electrodes       = self.get_particle_electrode_count()
            adv_index_rows, adv_index_cols  = self.get_advanced_indices()
            temperatures                    = self.get_const_temperatures(T=T_val)
            resistances                     = self.get_tunneling_rate_prefactor()

            # --- Instantiate (Numba-optimized) model ---
            model = monte_carlo.MonteCarlo(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                                temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles, floating_electrodes)

            # --- Run KMC simulation (with or without dynamic resistances) ---
            if self.dynamic_resistances:
                eq_jumps = model.run_equilibration_steps_var_resistance(
                    eq_steps, 
                    self.dynamic_resistances_info['slope'], 
                    self.dynamic_resistances_info['shift'],
                    self.dynamic_resistances_info['tau_0'],
                    self.dynamic_resistances_info['R_max'],
                    self.dynamic_resistances_info['R_min']
                )
                
                # Production Run until Current at target electrode is less than error_th or max_jumps was passed
                model.kmc_simulation_var_resistance(
                    target_electrode, error_th, max_jumps, jumps_per_batch, 
                    self.dynamic_resistances_info['slope'],
                    self.dynamic_resistances_info['shift'],
                    self.dynamic_resistances_info['tau_0'],
                    self.dynamic_resistances_info['R_max'],
                    self.dynamic_resistances_info['R_min'],
                    kmc_counting, verbose
                )
            else:
                eq_jumps = model.run_equilibration_steps(eq_steps)
                model.kmc_simulation(
                    target_electrode, error_th, max_jumps, jumps_per_batch,
                    output_potential, kmc_counting, min_batches, verbose
                )
            
            # --- Collect simulation results ---
            self.observable_storage.append(model.get_observable())
            self.observable_error_storage.append(model.get_observable_error())
            self.state_storage.append(model.get_state())
            self.potential_storage.append(model.get_potential())
            self.network_current_storage.append(model.get_network_current())
            self.eq_jump_storage.append(eq_jumps)
            self.jump_storage.append(model.get_jump())
            self.time_storage.append(model.get_time())

            # --- Store extra data if verbose ---
            if verbose:
                self.observable_per_batch.append(model.get_target_observable_per_it())
                self.time_per_batch.append(model.get_time_per_it())
                self.potential_per_batch.append(model.get_potential_per_it())
                # self.jump_per_batch.append(model.get_jump_per_batch())
                if self.dynamic_resistances:
                    self.resistance_per_batch.append(model.get_resistances_per_it())
                
            # --- Periodically save to disk ---
            if ((i + 1) % save_th == 0):
                self.data_to_path(voltages[j:(i + 1),:], self.path1)
                self.potential_to_path(self.path2)
                self.network_current_to_path(self.path3)
                self.clear_simulation_outputs()
                j = i+1

    def run_var_voltages(self, voltages : np.array, time_steps : np.array, target_electrode : int, T_val=0.0, eq_steps=0, save=True,
                         stat_size=10, init_charges=None, verbose=False):
        """Run a kinetic monte carlo simulation for time dependent electrode voltages to estimate either the electric current of
        the target electrode at variable target electrode voltage or the variable potential of a floating target electrode.

        Parameters
        ----------
        voltages : np.array
            2D-Array of electrode voltages with one time step per row, and columns defining the electrode indices.
        time_steps : np.array
            1D-Array defining the time scale for each voltage time step
        target_electrode : int
            Index of electrode for which the electric current or potential is estimated 
        T_val : float, optional
            Network temperature, by default 0.0
        eq_steps : int, optional
            Number of KMC steps for equilibration, by default 0
        save : bool, optional
            Store simulation results at path, by default True
        stat_size : int, optional
            Number of independet runs for statistics calculation, by default 500
        init_charges : np.array, optional
            Initialize network states and skip equilibration based 2D np.array with rows=Run and cols=NPs, by default None
        verbose : bool, optional
            If true, track additional simulation observables. Increases simulation duration!, by default False
        """

        # Round voltages to 0.01 mV
        voltages = np.round(voltages,5)
        
        # Identify floating electrodes and detect if target electrode is floating
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        const_electrodes    = np.where(self.electrode_type == 'constant')[0]
        output_potential    = (self.electrode_type[target_electrode] == 'floating')

        # Init based on first time step
        self.init_charge_vector(voltage_values=voltages[0])
        self.init_potential_vector(voltage_values=voltages[0])

        # Return Model Arguments
        inv_capacitance_matrix          = self.get_inv_capacitance_matrix()
        charge_vector                   = self.get_charge_vector()
        potential_vector                = self.get_potential_vector()
        const_capacitance_values        = self.get_const_capacitance_values()
        N_particles, N_electrodes       = self.get_particle_electrode_count()
        adv_index_rows, adv_index_cols  = self.get_advanced_indices()
        temperatures                    = self.get_const_temperatures(T=T_val)
        resistances                     = self.get_tunneling_rate_prefactor()
        
        # --- Instantiate (Numba-optimized) model ---
        self.model = monte_carlo.MonteCarlo(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                                temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles, floating_electrodes)

        # --- Run KMC simulation (with or without dynamic resistances) ---
        if init_charges is None:
            if self.dynamic_resistances:
                eq_jumps = self.model.run_equilibration_steps_var_resistance(
                        eq_steps, 
                        self.dynamic_resistances_info['slope'], 
                        self.dynamic_resistances_info['shift'],
                        self.dynamic_resistances_info['tau_0'],
                        self.dynamic_resistances_info['R_max'],
                        self.dynamic_resistances_info['R_min']
                    )
            else:
                eq_jumps = self.model.run_equilibration_steps(eq_steps)

        # Initial time
        self.model.time = time_steps[0]
        
        # Subtract charges induces by initial electrode voltages
        offset                      = self.get_charge_vector_offset(voltage_values=voltages[0])
        self.model.charge_vector    = self.model.charge_vector - offset
        
        # Without predefined states
        if init_charges is None:
            self.q_eq   = np.tile(self.model.charge_vector.copy(), (stat_size,1))
        else:
            eq_jumps    = 0
            self.q_eq   = init_charges

        # Reset Obseravles
        self.observable_storage         = np.zeros(voltages.shape[0])
        # self.output_values              = np.zeros(shape=(voltages.shape[0], 4))
        self.state_storage              = np.zeros(shape=(voltages.shape[0], self.model.N_particles))
        self.potential_storage          = np.zeros(shape=(voltages.shape[0], self.model.N_particles+self.model.N_electrodes))
        self.network_current_storage    = np.zeros(shape=(voltages.shape[0], len(self.model.adv_index_rows)))

        # Store equilibrated charge distribution
        observable = np.zeros(shape=(stat_size, len(voltages)))
        
        for s in range(stat_size):
            
            self.model.charge_vector = self.q_eq[s,:].copy()

            # For each time step, i.e. voltage
            for i, voltage_values in enumerate(voltages[:-1]):

                # Add charging state electrode voltage
                offset                      = self.get_charge_vector_offset(voltage_values=voltage_values)
                self.model.charge_vector    = self.model.charge_vector + offset
                
                # Define given time and time target
                self.model.time = time_steps[i]
                time_target     = time_steps[i+1]

                # Update Electrode Potentials
                self.model.potential_vector[const_electrodes] = voltage_values[const_electrodes]
                                
                if self.dynamic_resistances:
                    self.model.kmc_time_simulation_var_resistance(
                        target_electrode, time_target,
                        self.dynamic_resistances_info['slope'], 
                        self.dynamic_resistances_info['shift'],
                        self.dynamic_resistances_info['tau_0'],
                        self.dynamic_resistances_info['R_max'],
                        self.dynamic_resistances_info['R_min']
                    )
                else:
                    self.model.kmc_time_simulation(target_electrode, time_target, output_potential)
                    target_observable_mean  = self.model.get_observable()
                    total_jumps             = self.model.get_jump()
                
                # Add observables to outputs
                observable[s,i]                     =  target_observable_mean
                self.state_storage[i,:]             += self.model.get_state()/stat_size
                self.potential_storage[i,:]         += self.model.get_potential()/stat_size
                self.network_current_storage[i,:]   += self.model.get_network_current()/stat_size

                # Subtract past charging state voltage contribution
                self.model.charge_vector = self.model.charge_vector - offset

            # Last charge vector
            self.q_eq[s,:] = self.model.charge_vector.copy()

        # Calculate Average observable and error
        self.observable_storage         = np.mean(observable, axis=0)
        self.observable_error_storage   = 1.96*np.std(observable, axis=0, ddof=1)/np.sqrt(stat_size)

        # Delte last row
        self.state_storage              = np.delete(self.state_storage,-1, axis=0)
        self.potential_storage          = np.delete(self.potential_storage,-1, axis=0)
        self.network_current_storage    = np.delete(self.network_current_storage,-1, axis=0)

        V_safe_vals                         = np.zeros(shape=(self.potential_storage.shape[0],self.N_electrodes+1))
        V_safe_vals[:,floating_electrodes]  = self.potential_storage[:,floating_electrodes]
        V_safe_vals[:,const_electrodes]     = voltages[:-1, const_electrodes]
        V_safe_vals[:,-1]                   = voltages[:-1, -1]

        if save:
            output_pots = self.potential_storage[:,self.N_electrodes:]
            self.data_to_path(V_safe_vals, self.path1)
            self.potential_to_path(self.path2)
            self.network_current_to_path(self.path3)
        
    def clear_simulation_outputs(self) -> None:
        """Clears simulation outputs before running a new set of voltages."""
        
        # Default Data
        self.observable_storage         = []
        self.observable_error_storage   = []
        self.state_storage              = []
        self.potential_storage          = []
        self.network_current_storage    = []
        self.eq_jump_storage            = []
        self.jump_storage               = []
        self.time_storage               = []
        
        # Verbose Data
        self.observable_per_batch   = []
        # self.jump_per_batch         = []
        self.time_per_batch         = []
        self.potential_per_batch    = []
        self.resistance_per_batch   = []

    def get_observable_storage(self) -> np.ndarray:
        return np.array(self.observable_storage)
    
    def get_state_storage(self) -> np.ndarray:
        return np.array(self.state_storage)
    
    def get_potential_storage(self) -> np.ndarray:
        return np.array(self.potential_storage)
    
    def get_network_current_storage(self) -> dict:
        return {(self.adv_index_rows[i], self.adv_index_cols[i]) : val for i, val in enumerate(self.network_current_storage)}
    
    def get_eq_jump_storage(self) -> np.ndarray:
        return np.array(self.eq_jump_storage)
    
    def get_jump_storage(self) -> np.ndarray:
        return np.array(self.jump_storage)
    
    def get_time_storage(self) -> np.ndarray:
        return np.array(self.time_storage)
    
    def get_observable_per_batch(self) -> np.ndarray:
        return np.array(self.observable_per_batch)
    
    def get_jump_per_batch(self) -> np.ndarray:
        return np.array(self.jump_per_batch)

    def get_time_per_batch(self) -> np.ndarray:
        return np.array(self.time_per_batch)
    
    def get_potential_per_batch(self) -> np.ndarray:
        return np.array(self.potential_per_batch)
    
    def get_resistance_per_batch(self) -> dict:
        return {(self.adv_index_rows[i], self.adv_index_cols[i]) : val for i, val in enumerate(self.resistance_per_batch)}
    
    def data_to_path(self, voltages : np.ndarray, path : str) -> None:
        
        data    = np.hstack((voltages, [self.eq_jump_storage, self.jump_storage, self.observable_storage, self.observable_error_storage]))
        columns = [f'E{i}' for i in range(voltages.shape[1]-1)]
        columns = np.array(columns + ['G', 'Eq_Jumps', 'Jumps', 'Observable', 'Error'])

        df          = pd.DataFrame(data)
        df.columns  = columns

        if (os.path.isfile(path)):
            df.to_csv(path, mode='a', header=False, index=False)
        else:
            df.to_csv(path, header=True, index=False)

    def potential_to_path(self, path: str) -> None:

        microstates_df = pd.DataFrame(self.potential_storage)

        if (os.path.isfile(path)):
            microstates_df.to_csv(path, mode='a', header=False, index=False)
        else:
            microstates_df.to_csv(path, header=True, index=False)

    def network_current_to_path(self, path : str) -> None:

        avg_j_cols                  = [(self.adv_index_rows[i],self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))]
        average_jumps_df            = pd.DataFrame(self.average_jumps)
        average_jumps_df.columns    = avg_j_cols
        average_jumps_df            = average_jumps_df

        if (os.path.isfile(path)):
            average_jumps_df.to_csv(path, mode='a', header=False, index=False)
        else:
            average_jumps_df.to_csv(path, header=True, index=False)