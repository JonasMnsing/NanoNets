import numpy as np
from typing import Tuple
from numba.experimental import jitclass
from numba import int64, float64, boolean

# --- JIT Type Spec
spec = [
    ('charge_vector', float64[::1]),              # Charges per nanoparticle [aC]
    ('potential_vector', float64[::1]),           # Potentials [V]
    ('tunnel_rates', float64[::1]),               # Tunneling rates [1/s]
    ('inv_capacitance_matrix', float64[:,::1]),   # Inverse capacitance [1/aF]
    ('const_capacitance_values', float64[::1]),   # Capacitance terms for delta F [aC^2/aF]
    ('floating_electrodes', int64[::1]),          # Floating electrode indices
    ('temperatures', float64[::1]),               # Temperature for each tunnel event [aJ]
    ('resistances', float64[::1]),                # Resistances per tunneling event [MΩ]
    ('adv_index_rows', int64[::1]),               # Origin indices (i) for events i→j
    ('adv_index_cols', int64[::1]),               # Target indices (j) for events i→j
    ('N_electrodes', int64),                      # Number of electrodes
    ('N_particles', int64),                       # Number of nanoparticles
    ('ele_charge', float64),                      # Elementary charge [aC]
    ('planck_const', float64),                    # Reduced Planck constant [aJ·s]
    ('time', float64),                            # KMC time [s]
    ('counter_output_jumps_pos', float64),        # # of jumps *to* target electrode
    ('counter_output_jumps_neg', float64),        # # of jumps *from* target electrode
    ('target_observable_error_rel', float64),     # Rel. error of target observable
    ('target_observable_mean', float64),          # Mean of target observable
    ('target_observable_mean2', float64),         # Helper for error calc
    ('target_observable_error', float64),         # Std of target observable
    ('time_values', float64[::1]),                # List of time values
    ('total_jumps', int64),                       # Total KMC jumps
    ('jump', int64),                              # Last jump event index
    ('zero_T', boolean),                          # Use zero-T approx for rates
    ('charge_mean', float64[::1]),                # Mean charge (per NP)
    ('potential_mean', float64[::1]),             # Mean potential (per node)
    ('resistance_mean', float64[::1]),            # Mean resistance (per event)
    ('network_currents', float64[::1]),           # Current per junction
    ('I_tilde', float64[::1]),                    # Helper for memristor model
    ('resistances_per_it', float64[:,::1]),       # Resistances per iteration
    ('potential_per_it', float64[:,::1]),         # Potentials per iteration
    ('time_per_it', float64[::1]),                # Time values per iteration
    ('target_observable_per_it', float64[::1]),   # List of observable values
    ('N_rates', int64),                           # Number of tunneling events
]

@jitclass(spec)
class MonteCarlo():
    """
    Numba-optimized class for Kinetic Monte Carlo simulation of electron transport
    in nanoparticle networks. Efficiently simulates single-electron tunneling,
    computes electric currents, charge/potential landscapes, and supports
    network configurations with constant or floating electrodes.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_rates : int
        Number of tunneling events
    inv_capacitance_matrix : ndarray
        Inverse capacitance [1/aF]
    zero_T : bool
        Use zero-temperature approximation for rates.
    adv_index_rows : ndarray
        Origin nodes for all events.
    adv_index_cols : ndarray
        Target nodes for all events.
    potential_vector : ndarray
        Potentials [V] for all nodes.
    charge_vector : ndarray
        Charges [aC] for nanoparticles.
    tunnel_rates : ndarray
        Tunneling rates [1/s] for all events.
    const_capacitance_values : ndarray
        Capacitance terms for ΔF calculation [aC^2/aF].
    temperatures : ndarray
        Event-wise temperature [aJ].
    resistances : ndarray
        Resistances [MΩ] for all junctions.
    counter_output_jumps_pos : float
        # of jumps *to* target electrode
    counter_output_jumps_neg : float
        # of jumps *from* target electrode
    total_jumps : int
        Total KMC jumps
    time : float
        KMC time [s]
    target_observable_mean : float
        Mean of target observable
    target_observable_error : float
        Std of target observable
    target_observable_error_rel : float
        Rel. error of target observable
    jump : int
        Last jump event index
    charge_mean : ndarray
        Mean charge (per NP)
    potential_mean : ndarray
        Mean potential (per node)
    I_network : ndarray
        Current per junction
    I_tilde : ndarray
        Helper for memristor model
    """

    def __init__(self, charge_vector: np.ndarray, potential_vector: np.ndarray, inv_capacitance_matrix: np.ndarray, const_capacitance_values: np.ndarray,
                 temperatures: np.ndarray, resistances: np.ndarray, adv_index_rows: np.ndarray, adv_index_cols: np.ndarray, N_electrodes: int,
                 N_particles: int, floating_electrodes: np.ndarray) -> None:
        """
        Initialize the KMC simulation state and all model parameters.

        Parameters
        ----------
        charge_vector : ndarray
            Initial charge values per nanoparticle [aC].
        potential_vector : ndarray
            Initial potentials [V] for all nodes.
        inv_capacitance_matrix : ndarray
            Inverse of capacitance matrix [1/aF].
        const_capacitance_values : ndarray
            Capacitance terms for free energy calc [aC^2/aF].
        temperatures : ndarray
            Temperatures [aJ] for each tunneling event.
        resistances : ndarray
            Tunnel resistances [MΩ] for each event.
        adv_index_rows : ndarray
            Event origins (indices for i in i→j).
        adv_index_cols : ndarray
            Event targets (indices for j in i→j).
        N_electrodes : int
            Number of electrodes.
        N_particles : int
            Number of nanoparticles.
        floating_electrodes : ndarray
            Indices of floating electrodes.
        """
        # Physical constants
        self.ele_charge     = 0.160217662       # [aC]
        self.planck_const   = 1.054571817e-16   # [aJ·s]

        # System configuration
        self.charge_vector                  = charge_vector
        self.potential_vector               = potential_vector
        self.inv_capacitance_matrix         = inv_capacitance_matrix
        self.const_capacitance_values       = const_capacitance_values
        self.floating_electrodes            = floating_electrodes
        self.temperatures                   = temperatures
        self.resistances                    = resistances
        self.adv_index_rows                 = adv_index_rows
        self.adv_index_cols                 = adv_index_cols
        self.N_electrodes                   = N_electrodes
        self.N_particles                    = N_particles
        self.N_rates                        = len(self.adv_index_rows)

        # Simulation state variables
        self.counter_output_jumps_pos       = 0
        self.counter_output_jumps_neg       = 0
        self.total_jumps                    = 0
        self.time                           = 0.0
        self.target_observable_mean         = 0.0
        self.target_observable_mean2        = 0.0
        self.target_observable_error        = 0.0
        self.target_observable_error_rel    = 1.0
        self.jump                           = 0

        # Output accumulators
        self.charge_mean        = np.zeros(len(charge_vector))
        self.potential_mean     = np.zeros(len(potential_vector))
        self.network_currents   = np.zeros(len(adv_index_rows))
        self.resistance_mean    = np.zeros(len(resistances))
        self.I_tilde            = np.zeros(len(adv_index_rows))

        # Simulation control
        self.zero_T = False
        if np.sum(self.temperatures) == 0.0:
            self.zero_T = True

    def calc_potentials(self) -> None:
        """
        Compute updated nanoparticle potentials from current charge state.

        Updates the nanoparticle entries of self.potential_vector (indices N_electrodes:).
        Potentials are calculated as:
            V_NP = inv_capacitance_matrix @ charge_vector
        Units: [V] = [1/aF] * [aC]
        """
        self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)
    
    def update_floating_electrode(self, idx_np_target: np.ndarray) -> None:
        """
        Update potentials of floating electrodes based on adjacent nanoparticle potentials.

        Parameters
        ----------
        idx_np_target : np.ndarray
            Indices of nanoparticles adjacent to each floating electrode.

        Notes
        -----
        - Sets the potential of each floating electrode equal to the potential of its attached nanoparticle.
        - Typically called after nanoparticle potentials are updated.
        """
        self.potential_vector[self.floating_electrodes] = self.potential_vector[idx_np_target]
        
    def calc_tunnel_rates(self):
        """
        Compute tunneling rates for all possible tunneling events (finite T only).
        Uses the orthodox theory for electron tunneling.
        """
        # Calculate energy difference for tunneling events
        free_energy = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values

        # Numerically safe denominator
        exp_arg = free_energy / self.temperatures
        denom   = np.expm1(exp_arg)

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rates = (free_energy / self.resistances) / denom
            rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.tunnel_rates = rates

    def calc_tunnel_rates_zero_T(self):
        """
        Compute tunneling rates using the zero-temperature (T=0) approximation.
        
        In this regime, only energetically favorable events (ΔF < 0) occur.
        Rate = -ΔF/R if ΔF < 0, else 0.
        """
        # Calculate free energy change for all tunneling events
        free_energy = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        
        # Initialize all rates to zero
        self.tunnel_rates = np.zeros(self.N_rates)

        # Find allowed transitions (ΔF < 0)
        allowed = free_energy < 0

        # Only these events get a nonzero rate: -ΔF/R
        self.tunnel_rates[allowed] = -free_energy[allowed] / self.resistances[allowed]

    def calc_rel_error(self, N_calculations : int):
        """
        Calculate relative error and standard deviation for the target observable.
        
        Uses Welford's online algorithm for calculating variance in one pass.

        Parameters
        ----------
        N_calculations : int
            Number of measurements/calculations performed
        """
        if N_calculations >= 2:
            # Standard error (95% CI, 1.96)
            self.target_observable_error = 1.96*np.sqrt(
                np.abs(self.target_observable_mean2) / (N_calculations - 1)
                ) / np.sqrt(N_calculations)
            
            # Calculate relative error
            mean = np.abs(self.target_observable_mean)
            if mean > 0:
                self.target_observable_error_rel = self.target_observable_error/mean
            else:
                self.target_observable_error_rel = np.nan
    
    def return_next_means(self, new_value: float, mean_value: float, mean_value2: float, count: int) -> Tuple[float, float, int]:
        """
        Update running mean and variance (Welford's algorithm).

        Parameters
        ----------
        new_value : float
            New observation value.
        mean_value : float
            Current running mean.
        mean_value2 : float
            Current sum of squares of differences from the mean ("M2").
        count : int
            Number of values seen so far.

        Returns
        -------
        Tuple[float, float, int]
            Updated mean, M2, and count. To get sample variance: var = M2 / (count - 1) (for count > 1).
        """
        count       +=  1
        delta       =   new_value - mean_value
        mean_value  +=  delta / count          
        delta2      =   new_value - mean_value
        mean_value2 +=  delta * delta2

        return mean_value, mean_value2, count
    
    def select_event(self, random_number1 : float, random_number2 : float):
        """
        Select and apply the next tunneling event using kinetic Monte Carlo (KMC).

        Parameters
        ----------
        random_number1 : float
            Uniform random number (0,1) for selecting which event occurs.
        random_number2 : float
            Uniform random number (0,1) for time increment.
        """
        # Calculate cumulative sum of tunnel rates
        kmc_cum_sum = np.cumsum(self.tunnel_rates)
        k_tot       = kmc_cum_sum[-1]

        # No available events: do nothing, mark invalid jump
        if k_tot == 0.0:
            self.jump = -1
            return
        
        # Pick event: inverse transform sampling
        event   = random_number1 * k_tot
        jump    = np.searchsorted(a=kmc_cum_sum, v=event)   

        # Find indices for the jump
        np1 = self.adv_index_rows[jump]
        np2 = self.adv_index_cols[jump]

        # Identify source/destination types
        is_np1_electrode = (np1 - self.N_electrodes) < 0
        is_np2_electrode = (np2 - self.N_electrodes) < 0

        if is_np1_electrode: # Electrode -> NP
            self.charge_vector[np2 - self.N_electrodes] += self.ele_charge
            self.potential_vector[self.N_electrodes:]   += self.ele_charge * self.inv_capacitance_matrix[:, np2 - self.N_electrodes]
        elif is_np2_electrode: # NP -> Electrode
            self.charge_vector[np1 - self.N_electrodes] -= self.ele_charge
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge * self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
        else: # NP -> NP
            self.charge_vector[np1 - self.N_electrodes] -= self.ele_charge
            self.charge_vector[np2 - self.N_electrodes] += self.ele_charge
            delta_potential                             = self.inv_capacitance_matrix[:, np2 - self.N_electrodes] - self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
            self.potential_vector[self.N_electrodes:]   += self.ele_charge * delta_potential

        # Advance time: classic KMC time step
        self.time += -np.log(random_number2) / k_tot
        self.jump = jump

    def neglect_last_event(self, np1: int, np2: int):
        """
        Reverse the last tunneling event to restore the previous system state.

        Parameters
        ----------
        np1 : int
            Origin particle/electrode index for the event to reverse
        np2 : int
            Target particle/electrode index for the event to reverse
        """
        is_np1_electrode = (np1 - self.N_electrodes) < 0
        is_np2_electrode = (np2 - self.N_electrodes) < 0

        # If Electrode is involved
        if is_np1_electrode: # Undo: Electrode -> NP
            self.charge_vector[np2-self.N_electrodes]   -= self.ele_charge
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge*self.inv_capacitance_matrix[:,np2-self.N_electrodes]
        elif is_np2_electrode: # Undo: NP -> Electrode 
            self.charge_vector[np1-self.N_electrodes]   += self.ele_charge
            self.potential_vector[self.N_electrodes:]   += self.ele_charge*self.inv_capacitance_matrix[:,np1-self.N_electrodes]
        else: # Undo: NP -> NP 
            self.charge_vector[np1-self.N_electrodes]   += self.ele_charge
            self.charge_vector[np2-self.N_electrodes]   -= self.ele_charge
            delta_potential                             = self.inv_capacitance_matrix[:, np2 - self.N_electrodes] - self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge * delta_potential
        
    def run_equilibration_steps(self, n_jumps: int = 10000) -> int:
        """
        Execute a fixed number of KMC steps to equilibrate the system.

        Parameters
        ----------
        n_jumps : int, optional
            Number of KMC steps to execute, by default 10000

        Returns
        -------
        int
            Number of executed KMC steps
        """
        # Get indices of nanoparticles adjacent to floating electrodes
        idx_np_target = self.adv_index_cols[self.floating_electrodes]

        # Initial potentials
        self.calc_potentials()
        self.update_floating_electrode(idx_np_target)

        steps = 0
        for i in range(n_jumps):
            # Check if previous step was not valid
            if (self.jump == -1):
                break
            
            # Generate random numbers for KMC step
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            # Update tunneling rates
            if not self.zero_T:
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()
            
            # KMC event selection and state update
            self.select_event(random_number1, random_number2)
            self.update_floating_electrode(idx_np_target)
            steps += 1
                        
        return steps
    
    def run_equilibration_steps_var_resistance(self, n_jumps: int = 10000, slope: float = 0.8, 
                                              shift: float = 7.5, tau_0: float = 1e-8, R_max: float = 25, R_min: float = 10) -> int:
        """
        Execute a fixed number of KMC steps for equilibration with variable resistance (memristive behavior).

        Parameters
        ----------
        n_jumps : int, optional
            Number of KMC steps to execute, by default 10000
        slope : float, optional
            Memristor activation slope parameter, by default 0.8
        shift : float, optional
            Memristor activation shift parameter, by default 7.5
        tau_0 : float, optional
            Memory time constant, by default 1e-8
        R_max : float, optional
            Maximum resistance value, by default 25
        R_min : float, optional
            Minimum resistance value, by default 10

        Returns
        -------
        int
            Number of executed KMC steps
        """

        # Initialize current accumulator for memristive behavior
        self.I_tilde = np.zeros(len(self.adv_index_rows))

        # Initialize potentials
        self.calc_potentials()

        # Execute equilibration steps
        for i in range(n_jumps):
            
            # Update resistances based on memristive model
            self.update_bimodal_resistance(slope, shift, R_max, R_min)
            
            # Store current time
            t1 = self.time

            # Check if previous step was valid
            if self.jump == -1:
                break
            
            # Generate random numbers for KMC step
            random_number1 = np.random.rand()
            random_number2 = np.random.rand()

            # Calculate tunneling rates based on temperature model
            if not self.zero_T:
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()
                        
            # Execute KMC step
            self.select_event(random_number1, random_number2)
                        
            # Calculate time delta
            t2 = self.time
            dt = t2 - t1

            # Update current accumulator with exponential decay
            self.I_tilde            = self.I_tilde * np.exp(-dt / tau_0)
            self.I_tilde[self.jump] += 1

        return n_jumps
        
    def kmc_simulation(self, target_electrode: int, error_th: float = 0.05, max_jumps: int = 10000000, jumps_per_batch: int = 5000,
                       output_potential: bool = False, kmc_counting: bool = False, min_batches: int = 10, verbose: bool = False):
        """
        Run kinetic Monte Carlo simulation until the target observable reaches the desired
        relative error or the maximum number of steps is exceeded. Statistics are batched.

        Parameters
        ----------
        target_electrode : int
            Electrode index for observable tracking.
        error_th : float, optional
            Desired relative error, by default 0.05.
        max_jumps : int, optional
            Maximum KMC steps, by default 10_000_000.
        jumps_per_batch : int, optional
            Number of KMC steps per batch, by default 5_000.
        output_potential : bool, optional
            If True, track target electrode potential as observable (else, track current).
        kmc_counting : bool, optional
            If True, compute current by counting jumps. If False, use tunnel rates.
        min_batches : int, optional
            Minimum number of batches required to terminate.
        verbose : bool, optional
            If True, track and store additional observables.

        Notes
        -----
        - Batch statistics use Welford's algorithm for mean and variance.
        - Verbose arrays are preallocated to max possible batches. Only first N are used.
        """
        # Reset all statistics
        self.total_jumps                    = 0
        self.target_observable_error_rel    = 1.0
        self.target_observable_mean         = 0.0
        self.target_observable_mean2        = 0.0
        self.target_observable_error        = 0.0

        # NP indices adjacent to floating electrodes
        idx_np_target = self.adv_index_cols[self.floating_electrodes]

        # Initialize storage arrays
        self.charge_mean        = np.zeros_like(self.charge_vector)
        self.potential_mean     = np.zeros_like(self.potential_vector)
        self.network_currents   = np.zeros_like(self.adv_index_rows)

        # For rate-based current: find tunnel event indice
        if not kmc_counting and not output_potential:
            rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
            rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]

        # Verbose mode: Preallocate storage for batch observable
        if verbose:
            max_batches                     = int(max_jumps // jumps_per_batch) + 1
            self.target_observable_per_it   = np.zeros(max_batches)
            self.potential_per_it           = np.zeros((max_batches, len(self.potential_vector)))
            self.time_per_it                = np.zeros(max_batches)

        # Tracking
        count       = 0     # Number of completed batches
        below_rel   = 0     # Number of consecutive batches below error threshold
        time_total  = 0.0   # Total simulation time

        # Initial potential landscape
        self.calc_potentials()
        self.update_floating_electrode(idx_np_target)

        while (below_rel < min_batches) and (self.total_jumps < max_jumps):

            # Initialize batch counters
            if kmc_counting:
                self.counter_output_jumps_pos = 0
                self.counter_output_jumps_neg = 0
            else:
                target_value = 0.0

            # Initialize batch storage arrays
            time_values         = np.zeros(jumps_per_batch)
            charge_values       = np.zeros(self.N_particles)
            potential_values    = np.zeros(self.N_particles + self.N_electrodes)
            current_values      = np.zeros_like(self.adv_index_rows)
            self.time           = 0.0
            
            # --- Main batch loop ---
            for i in range(jumps_per_batch):
                # Record start time
                t1 = self.time
            
                # Draw random numbers
                random_number1 = np.random.rand()
                random_number2 = np.random.rand()

                # Calculate tunneling rates
                if not self.zero_T:
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()

                # For rate-based current, record current rates before jump
                if not kmc_counting and not output_potential:
                    rate1 = self.tunnel_rates[rate_index1]
                    rate2 = self.tunnel_rates[rate_index2]
                
                # Execute KMC step
                self.select_event(random_number1, random_number2)
                self.update_floating_electrode(idx_np_target)

                if self.jump == -1:
                    # No more possible events; exit batch
                    break

                # Get origin and destination for this jump
                np1 = self.adv_index_rows[self.jump]
                np2 = self.adv_index_cols[self.jump]
                
                # Record end time and time delta
                t2              = self.time
                dt              = t2 - t1
                time_values[i]  = dt

                # Weighted sum for averages
                charge_values       += self.charge_vector * dt
                potential_values    += self.potential_vector * dt
                current_values      += self.tunnel_rates * dt

                # Observable: output potential or output current
                if output_potential:
                    target_value += self.potential_vector[target_electrode] * dt
                else:
                    if kmc_counting:
                        # Count jumps to/from target electrode
                        if np1 == target_electrode:
                            self.counter_output_jumps_neg += 1
                        if np2 == target_electrode:
                            self.counter_output_jumps_pos += 1
                    else:
                        # Use rate difference
                        target_value += (rate1 - rate2) * dt

            # --- Batch post-processing ---
            self.total_jumps    += i + 1
            time_total          += self.time

            if self.time > 0:
                # Update time-weighted means
                self.charge_mean        += charge_values
                self.potential_mean     += potential_values
                self.network_currents   += current_values

                if kmc_counting and not output_potential:
                    target_observable = (self.counter_output_jumps_pos - self.counter_output_jumps_neg) / self.time
                else:
                    target_observable = target_value / self.time

                if verbose:
                    if output_potential:
                        self.target_observable_per_it[count]    = target_observable
                    else:
                        self.target_observable_per_it[count]    = self.ele_charge * target_observable
                    self.time_per_it[count]                 = self.time
                    self.potential_per_it[count,:]          = potential_values / self.time
                
                # Update running means and network currents 
                self.target_observable_mean, self.target_observable_mean2, count    = self.return_next_means(
                    target_observable, self.target_observable_mean, self.target_observable_mean2, count)

                # Calculate relative error
                if (self.target_observable_mean != 0):
                    self.calc_rel_error(count)

                if self.target_observable_error_rel < error_th:
                    below_rel += 1
                else:
                    below_rel = 0

            else: # If time didn't advance
                self.charge_mean        += self.charge_vector
                self.potential_mean     += self.potential_vector
                self.network_currents   += self.tunnel_rates

                if output_potential:
                    self.target_observable_mean = self.potential_vector[target_electrode]
                else:
                    self.target_observable_mean = 0.0
            
            # Catch invalid step
            if (self.jump == -1):
                break
        
        if not output_potential:
            self.target_observable_mean  *= self.ele_charge
            self.target_observable_error *= self.ele_charge

        # Final averaging
        if ((count != 0) and (self.jump != -1)):
            self.charge_mean        = self.charge_mean / time_total
            self.potential_mean     = self.potential_mean / time_total
            self.network_currents   = self.ele_charge * self.network_currents / time_total

    def kmc_time_simulation(self, target_electrode : int, time_target : float, output_potential : bool = True):
        """
        Run kinetic Monte Carlo simulation up to a target simulation time.

        Tracks output potential (default) or current, as well as time-averaged
        charge and potential landscapes. Network currents are accumulated as events.

        Parameters
        ----------
        target_electrode : int
            Electrode index for which to estimate observable (potential or current).
        time_target : float
            Target simulation time.
        output_potential : bool, optional
            If True, track output electrode potential (default). If False, track current.
        """
        # Calculate initial potential landscape
        self.calc_potentials()
        
        # Initialize storage arrays
        self.charge_mean        = np.zeros_like(self.charge_vector)
        self.potential_mean     = np.zeros_like(self.potential_vector)
        self.network_currents   = np.zeros_like(self.adv_index_rows)
        self.total_jumps        = 0

        # Indices for floating electrode updates
        idx_np_target = self.adv_index_cols[self.floating_electrodes]
        self.update_floating_electrode(idx_np_target)

        # Track simulation times        
        inner_time  = self.time
        last_time   = 0.0

        if not output_potential:
            # Identify target electrode rate indices
            rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
            rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]

        # Target observable
        target_value = 0.0  

        # --- Main simulation loop ---
        while (self.time < time_target):
            # Record current time before the event
            last_time   = self.time

            # Generate random numbers for KMC step
            random_number1 = np.random.rand()
            random_number2 = np.random.rand()

            # Calculate tunneling rates based on temperature model
            if not self.zero_T:
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()

            if not output_potential:
                # Record rate difference for target electrode
                rate1 = self.tunnel_rates[rate_index1]
                rate2 = self.tunnel_rates[rate_index2]

            # KMC Step and evolve in time
            self.select_event(random_number1, random_number2)

            # If system blocked or done, update for remaining time and exit
            if (self.jump == -1):
                self.update_floating_electrode(idx_np_target)
                dt = time_target - last_time
                if output_potential:
                    target_value    += self.potential_vector[target_electrode] * dt
                else:
                    target_value    += (rate1 - rate2) * dt
                self.charge_mean        += self.charge_vector * dt
                self.potential_mean     += self.potential_vector * dt
                self.network_currents   += self.tunnel_rates * dt
                break

            # Get origin and destination indices
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time, roll back the last event
            if self.time >= time_target:
                self.neglect_last_event(np1,np2)
                self.update_floating_electrode(idx_np_target)
                dt = time_target - last_time
                if output_potential:
                    target_value    += self.potential_vector[target_electrode] * dt
                else:
                    target_value    += (rate1-rate2)*dt
                self.charge_mean        += self.charge_vector * dt
                self.potential_mean     += self.potential_vector * dt
                self.network_currents   += self.tunnel_rates * dt
                break

            else:
                self.update_floating_electrode(idx_np_target)
                dt = self.time - last_time
                if output_potential:
                    target_value    += self.potential_vector[target_electrode]*dt
                else:
                    target_value    += (rate1-rate2)*dt

                self.charge_mean        += self.charge_vector*dt
                self.potential_mean     += self.potential_vector*dt
                self.network_currents   += self.tunnel_rates * dt
                self.total_jumps        += 1           
        
        # --- Final averages ---
        total_sim_time              = time_target - inner_time
        if output_potential:
            self.target_observable_mean = target_value / total_sim_time
        else:
            self.target_observable_mean = self.ele_charge * target_value / total_sim_time
        self.charge_mean            = self.charge_mean / total_sim_time
        self.potential_mean         = self.potential_mean / total_sim_time
        self.network_currents       = self.ele_charge * self.network_currents / total_sim_time

    # TODO: Check     
    def kmc_time_simulation_var_resistance(self, target_electrode : int, time_target : float, slope=0.8,
                            shift=7.5, tau_0=1e-8, R_max=25, R_min=10):
        """
        Runs KMC until KMC time exceeds a target value

        Parameters
        ----------
        target_electrode : int
            electrode index of which electric current is estimated
        time_target : float
            time value to be reached
        """
        
        self.total_jumps        = 0
        self.charge_mean        = np.zeros(len(self.charge_vector))
        self.resistance_mean    = np.zeros(len(self.resistances))
        self.potential_mean     = np.zeros(len(self.potential_vector))
        self.I_network          = np.zeros(len(self.adv_index_rows))

        inner_time  = self.time
        last_time   = 0.0
        rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
        rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]
        rate_diffs  = 0.0

        while (self.time < time_target):

            # Start time
            last_time   = self.time

            # KMC Part
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            # T=0 Approximation of Rates
            if not(self.zero_T):
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()

            # Extract rate difference
            rate1   = self.tunnel_rates[rate_index1]
            rate2   = self.tunnel_rates[rate_index2]

            # KMC Step and evolve in time
            self.select_event(random_number1, random_number2)

            if (self.jump == -1):
                break

            # Occured jump
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time
            if (self.time >= time_target):
                self.neglect_last_event(np1,np2)
                break

            # Current contribution to resistance
            self.I_tilde             = self.I_tilde*np.exp(-(self.time-last_time)/tau_0)
            self.I_tilde[self.jump]  += 1

            # New resistances
            self.update_bimodal_resistance(slope, shift, R_max, R_min)

            # Update Observables
            rate_diffs                  += (rate1 - rate2)*(self.time-last_time)
            self.charge_mean            += self.charge_vector*(self.time-last_time)
            self.resistance_mean        += self.resistances*(self.time-last_time)
            self.potential_mean         += self.potential_vector*(self.time-last_time)
            self.I_network[self.jump]   += 1            
            self.total_jumps            += 1
        
        if (self.jump == -1):
            self.target_observable_mean  = 0.0

        if (last_time-inner_time) != 0:
            
            self.target_observable_mean = rate_diffs/(time_target-inner_time)
            self.I_network              = self.I_network/(time_target-inner_time)
            self.charge_mean            = self.charge_mean/(time_target-inner_time)
            self.potential_mean         = self.potential_mean/(time_target-inner_time)
            self.resistance_mean        = self.resistance_mean/(time_target-inner_time)

        else:
            self.target_observable_mean = 0
            self.charge_mean            = self.charge_vector
            self.potential_mean         = self.potential_vector
            self.resistance_mean        = self.resistances

    # TODO: Check     
    def update_bimodal_resistance(self, slope : float, shift : float, R_max=30, R_min=20):

        r_max = R_max*self.ele_charge*self.ele_charge*1e-12
        r_min = R_min*self.ele_charge*self.ele_charge*1e-12

        self.resistances = (r_max - r_min)*(-np.tanh(slope*(self.I_tilde - shift)) + 1)/2 + r_min

    # TODO: Check     
    def kmc_simulation_var_resistance(self, target_electrode : int, error_th=0.05, max_jumps=10000000,
                                      jumps_per_batch=1000, slope=0.8, shift=7.5, tau_0=1e-8, 
                                      R_max=25, R_min=10, kmc_counting=True, verbose=False):
        """Runs kinetic Monte Carlo simulation until target electrode electric current reached desired relative error or
        maximum number of steps is exceeded. This algorithm calculates an electric current for fixed batches (number of jumps) and
        updates the average and standard error of the electric current according to those batches. The network resistances are variable.
        Whenever charges tunnel trough a particular junctions their resistances decrease, corresponding to a memristive behavior.
        
        Parameters
        ----------
        target_electrode : int
            Target electrode number
        error_th : float, optional
            Desired relative error in electric current, by default 0.05
        max_jumps : int, optional
            Maximum number of KMC steps, by default 10000000
        jumps_per_batch : int, optional
            Number of KMC steps per batch, by default 1000
        kmc_counting : bool, optional
            If True, electric current is calculated based on counting jumps. If False, based on tunnel rates, by default False
        verbose : bool, optional
            If True, simulation tracks additional observables, by default False
        """
        
        self.total_jumps                    = 0     # Total number of KMC Steps
        self.target_observable_error_rel    = 1.0   # Relative Error of I
        self.target_observable_mean         = 0.0   # Average Values of I
        self.target_observable_mean2        = 0.0   # Helper Value for I
        self.target_observable_error        = 0.0   # Standard Error of I

        self.charge_mean        = np.zeros(len(self.charge_vector))
        self.potential_mean     = np.zeros(len(self.potential_vector))
        self.I_network          = np.zeros(len(self.adv_index_rows))

        # If current based on tunnel rates, return target rate indices
        if kmc_counting == False:
            rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
            rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]
        
        # For addition informations, track batched electric currents
        if verbose:
            self.target_observable_values   = np.zeros(int(max_jumps/jumps_per_batch))
            self.time_values                = np.zeros(int(max_jumps/jumps_per_batch))
            self.landscape_per_it           = np.zeros((int(max_jumps/jumps_per_batch), len(self.potential_vector)))
            self.jump_dist_per_it           = np.zeros((int(max_jumps/jumps_per_batch), len(self.adv_index_rows)))
            self.resistances_per_it         = np.zeros((int(max_jumps/jumps_per_batch), len(self.adv_index_rows)))

        # # For additional information and without time scale bins
        # if (verbose and (jumps_per_batch == max_jumps)):

        #     self.landscape_per_it   = np.zeros((max_jumps, len(self.potential_vector)))
        #     self.resistances_per_it = np.zeros((max_jumps, len(self.adv_index_rows)))
        #     self.jump_dist_per_it   = np.zeros((max_jumps, len(self.adv_index_rows)))
        #     self.time_vals          = np.zeros(max_jumps)
        
        count       = 0     # Number of while loops
        time_total  = 0.0   # Total time passed

        # Calculate potential landscape once
        self.calc_potentials()        

        # While relative error or maximum amount of KMC steps not reached
        while(((self.target_observable_error_rel > error_th) and (self.total_jumps < max_jumps))):

            # Counting or not
            if kmc_counting:
                self.counter_output_jumps_pos   = 0
                self.counter_output_jumps_neg   = 0
            else:
                rate_diffs                      = 0.0

            # Reset Array to track occured jumps
            jump_storage_vals   = np.zeros(len(self.adv_index_rows))
            resistance_vals     = np.zeros(len(self.adv_index_rows))
            time_values         = np.zeros(jumps_per_batch)
            charge_values       = np.zeros(self.N_particles)
            potential_values    = np.zeros(self.N_particles+self.N_electrodes)
            self.time           = 0.0
            
            # KMC Run for a batch
            for i in range(jumps_per_batch):
                
                # Time before event
                t1  = self.time
                
                # Select an event based on rates
                random_number1  = np.random.rand()
                random_number2  = np.random.rand()

                # T=0 Approximation of Rates
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()

                # Without counting, extract rate difference
                if kmc_counting == False:
                    rate1   = self.tunnel_rates[rate_index1]
                    rate2   = self.tunnel_rates[rate_index2]
                
                # KMC Step and evolve in time
                self.select_event(random_number1, random_number2) 

                # Occured jump
                np1 = self.adv_index_rows[self.jump]
                np2 = self.adv_index_cols[self.jump]
                jump_storage_vals[self.jump]    += 1

                # Time after event
                t2              = self.time
                dt              = t2-t1
                time_values[i]  = dt

                # Add charge and potential vector
                charge_values       += self.charge_vector*(t2-t1)
                potential_values    += self.potential_vector*(t2-t1)
                
                # Current contribution to resistance
                self.I_tilde             = self.I_tilde*np.exp(-dt/tau_0)
                self.I_tilde[self.jump]  += 1

                # New resistances
                self.update_bimodal_resistance(slope, shift, R_max, R_min)
                resistance_vals += self.resistances*(t2-t1)

                if kmc_counting:
                    # If jump from target electrode
                    if (np1 == target_electrode):
                        self.counter_output_jumps_neg += 1
                        
                    # If jump towards target electrode
                    if (np2 == target_electrode):
                        self.counter_output_jumps_pos += 1
                else:
                    rate_diffs += (rate1 - rate2)*(t2-t1)
                
                # If additional informations are required, track observables
                # if (verbose and (jumps_per_batch == max_jumps)):

                #     self.landscape_per_it[i,:]      = self.potential_vector
                #     self.resistances_per_it[i,:]    = self.resistances
                #     self.jump_dist_per_it[i,:]      = jump_storage_vals
                #     self.time_vals[i,:]             = t2
                
            # Update total jumps, average charges, and average potentials
            self.total_jumps    += i+1
            self.charge_mean    += charge_values
            self.potential_mean += potential_values
            time_total          +=  self.time

            if self.time != 0:

                # Calc new average currents
                if kmc_counting:
                    target_observable   = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
                else:
                    target_observable   = rate_diffs/self.time

                if verbose:
                    self.target_observable_values[count]    = target_observable
                    self.time_values[count]                 = self.time
                    self.landscape_per_it[count,:]          = potential_values/self.time
                    self.jump_dist_per_it[count,:]          = jump_storage_vals
                    self.resistances_per_it[count,:]        = resistance_vals   

                self.target_observable_mean, self.target_observable_mean2, count    = self.return_next_means(target_observable, self.target_observable_mean, self.target_observable_mean2, count)
                self.I_network                                                      += jump_storage_vals/self.time

            if (self.jump == -1):
                self.target_observable_mean = 0.0
                break

            if (self.target_observable_mean != 0):
                self.calc_rel_error(count)

        if count != 0:
            self.I_network      = self.I_network/count
            self.charge_mean    = self.charge_mean/time_total
            self.potential_mean = self.potential_mean/time_total
      
    def get_observable(self):
        """
        Returns
        -------
        float
            Final time-averaged target observable (e.g., output current or potential) after the last simulation run.
        """
        return self.target_observable_mean
    
    def get_observable_error(self):
        """
        Returns
        -------
        float
            Estimated standard error of the target observable (computed over simulation batches).
        """
        return self.target_observable_error
    
    def get_state(self):
        """
        Returns
        -------
        np.ndarray
            Time-averaged nanoparticle charge state vector after simulation.
            Shape: (N_particles,)
        """
        return self.charge_mean
    
    def get_potential(self):
        """
        Returns
        -------
        np.ndarray
            Time-averaged network potential vector (electrodes + nanoparticles)
            after simulation. Shape: (N_electrodes + N_particles,)
        """
        return self.potential_mean
    
    def get_network_current(self):
        """
        Returns
        -------
        np.ndarray
            Time-averaged current for each tunneling event (i → j), units [1/time].
            Shape: (N_tunnel_events,)
        """
        return self.network_currents
    
    def get_jump(self):
        """
        Returns
        -------
        int
            Total number of accepted kinetic Monte Carlo steps (jumps)
            performed in the most recent simulation.
        """
        return self.total_jumps
    
    def get_time(self):
        """
        Returns
        -------
        float
            Total KMC simulation time elapsed in the last run.
        """
        return self.time
    
    def get_resistances_per_it(self):
        """
        Returns
        -------
        np.ndarray
            (If verbose mode) Resistance values recorded at each simulation batch.
            Shape: (N_batches, N_tunnel_events)
        """
        return self.resistances_per_it
    
    def get_potential_per_it(self):
        """
        Returns
        -------
        np.ndarray
            (If verbose mode) Batch-wise average potential landscapes.
            Shape: (N_batches, N_electrodes + N_particles)
        """
        return self.potential_per_it
    
    def get_time_per_it(self):
        """
        Returns
        -------
        np.ndarray
            (If verbose mode) Array of simulation time for each batch.
            Shape: (N_batches,)
        """
        return self.time_per_it
    
    def get_target_observable_per_it(self):
        """
        Returns
        -------
        np.ndarray
            (If verbose mode) Array of the target observable (current or potential)
            evaluated at each simulation batch. Shape: (N_batches,)
        """
        return self.target_observable_per_it