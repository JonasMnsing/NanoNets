import tunneling
import numpy as np
import pandas as pd
import os.path
from typing import Tuple, Optional, List, Union, Any
from numba.experimental import jitclass
from numba import int64, float64, boolean

# Define type specification with helpful comments
spec = [
    ('charge_vector', float64[::1]),              # Charge values for each nanoparticle
    ('potential_vector', float64[::1]),           # Potential values for electrodes and nanoparticles
    ('tunnel_rates', float64[::1]),               # Tunnel rate values for each tunneling event
    ('inv_capacitance_matrix', float64[:,::1]),   # Inverse of capacitance matrix
    ('const_capacitance_values', float64[::1]),   # Sum of capacitance for free energy calculation
    ('floating_electrodes', int64[::1]),          # Indices of floating electrodes
    ('temperatures', float64[::1]),               # Temperature values for each tunnel event
    ('resistances', float64[::1]),                # Resistances for each tunneling event i to j
    ('adv_index_rows', int64[::1]),               # Nanoparticles i (origin) in tunneling event i to j
    ('adv_index_cols', int64[::1]),               # Nanoparticles j (target) in tunneling event i to j
    ('N_electrodes', int64),                      # Number of electrodes
    ('N_particles', int64),                       # Number of nanoparticles
    ('ele_charge', float64),                      # Elementary charge constant
    ('planck_const', float64),                    # Planck's constant
    ('time', float64),                            # KMC time scale
    ('counter_output_jumps_pos', float64),        # Number of jumps towards target electrode
    ('counter_output_jumps_neg', float64),        # Number of jumps from target electrode
    ('target_observable_error_rel', float64),     # Relative error for difference in jumps
    ('target_observable_mean', float64),          # Mean difference in jumps towards/from target electrode
    ('target_observable_mean2', float64),         # Helper value for computing error
    ('target_observable_error', float64),         # Standard deviation for difference in jumps
    ('target_observable_values', float64[::1]),   # Storage for target observable values
    ('time_values', float64[::1]),                # Storage for time values
    ('total_jumps', int64),                       # Total number of jumps/KMC steps
    ('jump', int64),                              # Last occurred jump/event
    ('zero_T', boolean),                          # Flag for zero temperature approximation
    ('charge_mean', float64[::1]),                # Storage for average network charges
    ('potential_mean', float64[::1]),             # Storage for average potential landscape
    ('resistance_mean', float64[::1]),            # Storage for average resistances
    ('I_network', float64[::1]),                  # Network electric currents
    ('I_tilde', float64[::1]),                    # Helper array for memristive behavior
    ('jump_dist_per_it', float64[:,::1]),         # Storage for jumps per iteration
    ('resistances_per_it', float64[:,::1]),       # Storage for resistances per iteration
    ('landscape_per_it', float64[:,::1]),         # Storage for potential landscape per iteration
    ('time_vals', float64[::1]),                  # Storage for time values
    ('N_rates', int64),                           # Number of tunneling events
]

@jitclass(spec)
class model_class():
    """
    Numba-optimized class to run the Kinetic Monte Carlo (KMC) simulation for
    electron transport in nanoparticle networks.

    This class simulates electron tunneling events between nanoparticles and electrodes,
    and calculates electric currents, potentials, and charge distributions.

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles in the system
    N_electrodes : int
        Number of electrodes in the system
    N_rates : int
        Number of possible tunneling events
    inv_capacitance_matrix : ndarray
        Inverse of capacitance matrix used for potential calculations
    zero_T : bool
        Whether to use zero temperature approximation for tunneling rates
    adv_index_rows : ndarray
        Indices of origin particles/electrodes in tunneling events
    adv_index_cols : ndarray
        Indices of target particles/electrodes in tunneling events
    potential_vector : ndarray
        Potential values for electrodes and nanoparticles
    charge_vector : ndarray
        Charge values for each nanoparticle
    tunnel_rates : ndarray
        Tunneling rate values for each possible event
    const_capacitance_values : ndarray
        Sum of capacitances for free energy calculation
    temperatures : ndarray
        Temperature values for each tunnel event
    resistances : ndarray
        Resistance values for each tunneling event
    counter_output_jumps_pos : float
        Number of jumps towards target electrode
    counter_output_jumps_neg : float
        Number of jumps from target electrode
    total_jumps : int
        Total number of jumps/KMC steps executed
    time : float
        KMC simulation time
    target_observable_mean : float
        Mean value of the target observable (typically current)
    target_observable_error : float
        Standard deviation of the target observable
    target_observable_error_rel : float
        Relative error of the target observable
    jump : int
        Index of the last occurred tunneling event
    charge_mean : ndarray
        Average network charges over simulation time
    potential_mean : ndarray
        Average potential landscape over simulation time
    I_network : ndarray
        Contribution of each junction to the total current
    I_tilde : ndarray
        Helper array for memristive behavior simulation
    """

    def __init__(self, charge_vector: np.ndarray, potential_vector: np.ndarray, inv_capacitance_matrix: np.ndarray, const_capacitance_values: np.ndarray,
                 temperatures: np.ndarray, resistances: np.ndarray, adv_index_rows: np.ndarray, adv_index_cols: np.ndarray, N_electrodes: int,
                 N_particles: int, floating_electrodes: np.ndarray):
        """
        Initialize the KMC model with system parameters.

        Parameters
        ----------
        charge_vector : ndarray
            Initial charge values for each nanoparticle
        potential_vector : ndarray
            Initial potential values for electrodes and nanoparticles
        inv_capacitance_matrix : ndarray
            Inverse of capacitance matrix for potential calculations
        const_capacitance_values : ndarray
            Sum of capacitances for free energy calculation
        temperatures : ndarray
            Temperature values for each tunneling event
        resistances : ndarray
            Resistance values for each tunneling event
        adv_index_rows : ndarray
            Origin indices for tunneling events
        adv_index_cols : ndarray
            Target indices for tunneling events
        N_electrodes : int
            Number of electrodes in the system
        N_particles : int
            Number of nanoparticles in the system
        floating_electrodes : ndarray
            Indices of floating electrodes
        """
        # Physical constants
        self.ele_charge     = 0.160217662       # [aC]
        self.planck_const   = 1.054571817e-16   # Planck

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

        # Initialize storage arrays
        self.charge_mean        = np.zeros(len(charge_vector))
        self.potential_mean     = np.zeros(len(potential_vector))
        self.resistance_mean    = np.zeros(len(resistances))
        self.I_network          = np.zeros(len(adv_index_rows))
        self.I_tilde            = np.zeros(len(adv_index_rows))

        # Check if we're using zero temperature approximation
        self.zero_T = False
        if np.sum(self.temperatures) == 0.0:
            self.zero_T = True

    def calc_potentials(self):
        """
        Compute nanoparticle potentials using the inverse capacitance matrix.
        
        This method updates the potential_vector for nanoparticles (not electrodes)
        based on the current charge distribution.
        """

        self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)

    def update_potentials(self, np1 : int, np2 : int):
        """
        Update potentials after a tunneling event between particles/electrodes.

        Parameters
        ----------
        np1 : int
            Index of origin particle/electrode in the last tunneling event
        np2 : int
            Index of target particle/electrode in the last tunneling event
        """
        is_np1_electrode = (np1 - self.N_electrodes) < 0
        is_np2_electrode = (np2 - self.N_electrodes) < 0

        if is_np1_electrode and not is_np2_electrode:
            # Tunneling from electrode to nanoparticle
            self.potential_vector[self.N_electrodes:] += self.inv_capacitance_matrix[:, np2] * self.ele_charge
        elif is_np2_electrode and not is_np1_electrode:
            # Tunneling from nanoparticle to electrode
            self.potential_vector[self.N_electrodes:] -= self.inv_capacitance_matrix[:, np1] * self.ele_charge
        elif is_np1_electrode and is_np2_electrode:
            # Both are electrodes, recalculate all potentials
            self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)
        else:
            # Tunneling between nanoparticles
            delta_potential                             =   self.inv_capacitance_matrix[:, np2] - self.inv_capacitance_matrix[:, np1]
            self.potential_vector[self.N_electrodes:]   +=  delta_potential * self.ele_charge
    
    def update_floating_electrode(self, idx_np_target)->None:
        """Update potentials of floating electrodes based on adjacent nanoparticle potentials.

        Parameters
        ----------
        idx_np_target : np.array
            Indices of nanoparticles adjacent to floating electrodes  
        """
        self.potential_vector[self.floating_electrodes] = self.potential_vector[idx_np_target]
        
    def calc_tunnel_rates(self):
        """
        Compute tunneling rates for all possible tunneling events.
        
        The rates are calculated from orthodox tunneling theory using the free energy difference between origin
        and destination, temperature, and junction resistance.
        """

        # Calculate energy difference for tunneling events
        free_energy         = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values

        # Calculate tunneling rates using orthodox tunneling theory
        self.tunnel_rates   = (free_energy / self.resistances) / (np.exp(free_energy / self.temperatures) - 1.0)
        
        # Handle potential NaN values (e.g., at zero temperature or zero energy difference)
        self.tunnel_rates   = np.nan_to_num(self.tunnel_rates, nan=0.0)

    def calc_tunnel_rates_zero_T(self):
        """
        Compute tunneling rates using zero temperature approximation.
        
        In this approximation, tunneling only occurs when it's energetically favorable
        (free energy is negative).
        """
        # Calculate energy difference for tunneling events
        free_energy = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        
        # Initialize rates to zero
        self.tunnel_rates                   = np.zeros(self.N_rates)
        # Only events with negative free energy are possible
        self.tunnel_rates[free_energy<0]    = -free_energy[free_energy<0]/self.resistances[free_energy<0]

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
            # Calculate standard error using 95% confidence interval (1.96 factor)
            self.target_observable_error     = 1.96*np.sqrt(
                np.abs(self.target_observable_mean2) / (N_calculations - 1)
                )/np.sqrt(N_calculations)
            
            # Calculate relative error
            self.target_observable_error_rel = self.target_observable_error/np.abs(self.target_observable_mean)
    
    def select_event(self, random_number1 : float, random_number2 : float):
        """
        Select the next tunneling event using kinetic Monte Carlo approach and update the system state.
        
        Parameters
        ----------
        random_number1 : float
            Random number for selecting the event (0-1)
        random_number2 : float
            Random number for advancing time (0-1)
        """
        # Calculate cumulative sum of tunnel rates
        kmc_cum_sum = np.cumsum(self.tunnel_rates)
        k_tot       = kmc_cum_sum[-1]

        # If no tunneling events are possible, mark as invalid jump and return
        if k_tot == 0.0:
            self.jump = -1
            return
        
        # Select event based on random number weighted by rates
        event   = random_number1 * k_tot
        jump    = np.searchsorted(a=kmc_cum_sum, v=event)   

        # Get origin and destination indices
        np1 = self.adv_index_rows[jump]
        np2 = self.adv_index_cols[jump]

        # Update charge and potential based on the type of tunneling event
        is_np1_electrode = (np1 - self.N_electrodes) < 0
        is_np2_electrode = (np2 - self.N_electrodes) < 0

        if is_np1_electrode: # From electrode to nanoparticle
            self.charge_vector[np2 - self.N_electrodes] += self.ele_charge
            self.potential_vector[self.N_electrodes:]   += self.ele_charge * self.inv_capacitance_matrix[:, np2 - self.N_electrodes]
        elif is_np2_electrode: # From nanoparticle to electrode
            self.charge_vector[np1 - self.N_electrodes] -= self.ele_charge
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge * self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
        else: # Between nanoparticles
            self.charge_vector[np1 - self.N_electrodes] -= self.ele_charge
            self.charge_vector[np2 - self.N_electrodes] += self.ele_charge
            delta_potential                             = self.inv_capacitance_matrix[:, np2 - self.N_electrodes] - self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
            self.potential_vector[self.N_electrodes:]   += self.ele_charge * delta_potential

        # Update KMC time and track the last jump
        self.time += -np.log(random_number2) / k_tot
        self.jump = jump

    def neglect_last_event(self, np1: int, np2: int):
        """
        Reverse the last tunneling event to restore previous system state.

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
        if is_np1_electrode: # Reverse: electrode to nanoparticle
            self.charge_vector[np2-self.N_electrodes]   -= self.ele_charge
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge*self.inv_capacitance_matrix[:,np2-self.N_electrodes]
        elif is_np2_electrode: # Reverse: nanoparticle to electrode
            self.charge_vector[np1-self.N_electrodes]   += self.ele_charge
            self.potential_vector[self.N_electrodes:]   += self.ele_charge*self.inv_capacitance_matrix[:,np1-self.N_electrodes]
        else: # Reverse: nanoparticle to nanoparticle
            self.charge_vector[np1-self.N_electrodes]   += self.ele_charge
            self.charge_vector[np2-self.N_electrodes]   -= self.ele_charge
            delta_potential                             = self.inv_capacitance_matrix[:, np2 - self.N_electrodes] - self.inv_capacitance_matrix[:, np1 - self.N_electrodes]
            self.potential_vector[self.N_electrodes:]   -= self.ele_charge * delta_potential
        
    def run_equilibration_steps(self, n_jumps=10000):
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

        # Initialize potentials
        self.calc_potentials()
        self.update_floating_electrode(idx_np_target)

        # Execute equilibration steps
        for i in range(n_jumps):
            # Check if previous step was valid
            if (self.jump == -1):
                break
            
            # Generate random numbers for KMC step
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            # Calculate tunneling rates based on temperature model
            if not self.zero_T:
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()
            
            # Execute KMC step
            self.select_event(random_number1, random_number2)
            
            # Update floating electrode potentials
            self.update_floating_electrode(idx_np_target)
                        
        return n_jumps
    
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
    
    def return_next_means(self, new_value: float, mean_value: float, mean_value2: float, count: int) -> Tuple[float, float, int]:
        """
        Update running means and variance for online statistics calculation.
        
        Uses Welford's online algorithm for numerically stable mean/variance calculation.

        Parameters
        ----------
        new_value : float
            New observation value
        mean_value : float
            Current mean value
        mean_value2 : float
            Current M2 aggregate (sum of squared differences)
        count : int
            Current count of observations

        Returns
        -------
        Tuple[float, float, int]
            Updated mean, M2 aggregate, and count
        """
        count       +=  1
        delta       =   new_value - mean_value
        mean_value  +=  delta / count          
        delta2      =   new_value - mean_value
        mean_value2 +=  delta * delta2

        return mean_value, mean_value2, count
    
    def kmc_simulation(self, target_electrode : int, error_th = 0.05, max_jumps=10000000,
                             jumps_per_batch=5000, output_potential=False, kmc_counting=False, min_batches=10, verbose=False):
        """Runs kinetic Monte Carlo simulation until target observable reached desired relative error or
        maximum number of steps is exceeded. This algorithm calculates an observalbe for fixed batches (number of jumps) and
        updates the average and standard error of the observable according to those batches.     

        Parameters
        ----------
        target_electrode : int
            Electrode number associated to observable
        error_th : float, optional
            Desired relative error, by default 0.05
        max_jumps : int, optional
            Maximum number of KMC steps, by default 10000000
        jumps_per_batch : int, optional
            Number of KMC steps per batch, by default 1000
        output_potential : bool, optional
            If True, simulation tracks target electrode potential as observable. If False, simulation tracks target electrode electric current, by default False
        kmc_counting : bool, optional
            If True, electric current is calculated based on counting jumps. If False, based on tunnel rates, by default False
        min_batches : int, optional
            Minimum number of batches for statistics, by default 10
        verbose : bool, optional
            If True, simulation tracks additional observables, by default False
        """

        # Reset simulation statistics
        self.total_jumps                    = 0
        self.target_observable_error_rel    = 1.0
        self.target_observable_mean         = 0.0
        self.target_observable_mean2        = 0.0
        self.target_observable_error        = 0.0

        # Get indices of nanoparticles adjacent to floating electrodes
        idx_np_target = self.adv_index_cols[self.floating_electrodes]

        # Initialize storage arrays
        self.charge_mean    = np.zeros(len(self.charge_vector))
        self.potential_mean = np.zeros(len(self.potential_vector))
        self.I_network      = np.zeros(len(self.adv_index_rows))

        # If current based on tunnel rates, find target electrode rate indices
        if not kmc_counting:
            rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
            rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]

        # For additional information, allocate arrays to track batched values
        if verbose:
            max_batches                     = int(max_jumps / jumps_per_batch)
            self.target_observable_values   = np.zeros(max_batches)
            self.landscape_per_it           = np.zeros((max_batches, len(self.potential_vector)))
            self.time_values                = np.zeros(max_batches)
            self.jump_dist_per_it           = np.zeros((max_batches, len(self.adv_index_rows)))

        # Initialize tracking variables
        count       = 0     # Number of completed batches
        below_rel   = 0     # Number of consecutive batches below error threshold
        time_total  = 0.0   # Total simulation time

        # Calculate initial potential landscape
        self.calc_potentials()

        # Update floating electrode potentials
        self.update_floating_electrode(idx_np_target)

        # Main simulation loop
        while (below_rel < min_batches) and (self.total_jumps < max_jumps):

            # Initialize batch counters
            if kmc_counting:
                self.counter_output_jumps_pos = 0
                self.counter_output_jumps_neg = 0
            else:
                target_value = 0.0

            # Initialize batch storage arrays
            jump_storage_vals   = np.zeros(len(self.adv_index_rows))
            time_values         = np.zeros(jumps_per_batch)
            charge_values       = np.zeros(self.N_particles)
            potential_values    = np.zeros(self.N_particles + self.N_electrodes)
            self.time           = 0.0
            
            # Execute batch of KMC steps
            for i in range(jumps_per_batch):
                
                # Record start time
                t1 = self.time
            
                # Generate random numbers
                random_number1 = np.random.rand()
                random_number2 = np.random.rand()

                # Calculate tunneling rates
                if not self.zero_T:
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()

                # If not counting jumps directly, record rate difference
                if not kmc_counting:
                    rate1 = self.tunnel_rates[rate_index1]
                    rate2 = self.tunnel_rates[rate_index2]
                
                # Execute KMC step
                self.select_event(random_number1, random_number2)

                # Update floating electrode potentials
                self.update_floating_electrode(idx_np_target)

                # Check if step was valid
                if self.jump == -1:
                    break

                # Get origin and destination indices
                np1 = self.adv_index_rows[self.jump]
                np2 = self.adv_index_cols[self.jump]
                
                # Record jump occurrence
                jump_storage_vals[self.jump] += 1

                # Record end time and time delta
                t2              = self.time
                dt              = t2 - t1
                time_values[i]  = dt

                # Accumulate charge and potential weighted by time
                charge_values       += self.charge_vector * dt
                potential_values    += self.potential_vector * dt

                # Update observable based on simulation type
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

            # Update total jumps and time
            self.total_jumps    += i + 1
            time_total          += self.time

            # Process batch results if time has advanced
            if self.time != 0:
                # Accumulate charge and potential means
                self.charge_mean    += charge_values
                self.potential_mean += potential_values

                # Calculate observable value for this batch
                if kmc_counting and not output_potential:
                    target_observable = (self.counter_output_jumps_pos - self.counter_output_jumps_neg) / self.time
                else:
                    target_observable = target_value / self.time

                # Record verbose statistics if requested
                if verbose:
                    self.target_observable_values[count]    = target_observable
                    self.time_values[count]                 = self.time
                    self.landscape_per_it[count,:]          = potential_values/self.time
                    self.jump_dist_per_it[count,:]          = jump_storage_vals
                
                # Update running means and network currents 
                self.target_observable_mean, self.target_observable_mean2, count    = self.return_next_means(target_observable, self.target_observable_mean, self.target_observable_mean2, count)
                self.I_network                                                      += jump_storage_vals/self.time

                # Calculate the current relative error
                if (self.target_observable_mean != 0):
                    self.calc_rel_error(count)

                # Check if this batch is below relative error
                if self.target_observable_error_rel < error_th:
                    below_rel += 1
                else:
                    below_rel = 0

            else: # If time has not advanced
                self.charge_mean    += self.charge_vector
                self.potential_mean += self.potential_vector
                self.I_network      = jump_storage_vals/self.total_jumps

                if not output_potential:
                    self.target_observable_mean = 0.0
                else:
                    self.target_observable_mean = self.potential_vector[target_electrode]
            
            # Catch invalid step
            if (self.jump == -1):
                break
        
        # Final averages
        if ((count != 0) and (self.jump != -1)):
            self.I_network      = self.I_network/count
            self.charge_mean    = self.charge_mean/time_total
            self.potential_mean = self.potential_mean/time_total
     
    def kmc_time_simulation(self, target_electrode : int, time_target : float):
        """
        Runs KMC until KMC time exceeds a target value

        Parameters
        ----------
        target_electrode : int
            electrode index of which electric current is estimated
        time_target : float
            time value to be reached
        """

        # Calculate initial potential landscape
        self.calc_potentials()

        # Initialize storage arrays
        self.charge_mean    = np.zeros(len(self.charge_vector))
        self.potential_mean = np.zeros(len(self.potential_vector))
        self.I_network      = np.zeros(len(self.adv_index_rows))
        self.total_jumps    = 0

        # Batch times
        inner_time  = self.time
        last_time   = 0.0

        # Find target electrode rate indices
        rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
        rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]
        rate_diffs  = 0.0

        # Until we reach our time target
        while (self.time < time_target):

            # Start time
            last_time   = self.time

            # Generate random numbers
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            # Calculate tunneling rates
            if not self.zero_T:
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()

            # Record rate difference
            rate1   = self.tunnel_rates[rate_index1]
            rate2   = self.tunnel_rates[rate_index2]

            # Execute KMC step
            self.select_event(random_number1, random_number2)
            
            # Check if step was valid
            if (self.jump == -1):
                # Update Observables
                rate_diffs          += (rate1 - rate2)*(time_target-last_time)
                self.charge_mean    += self.charge_vector*(time_target-last_time)
                self.potential_mean += self.potential_vector*(time_target-last_time)
                break

            # Get origin and destination indices
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time
            if (self.time >= time_target):
                # Neglect last tunneling event
                self.neglect_last_event(np1,np2)
                # Update Observables
                rate_diffs                  += (rate1 - rate2)*(time_target-last_time)
                self.charge_mean            += self.charge_vector*(time_target-last_time)
                self.potential_mean         += self.potential_vector*(time_target-last_time)
                break

            else:
                # Update Observables
                rate_diffs                  += (rate1 - rate2)*(self.time-last_time)
                self.charge_mean            += self.charge_vector*(self.time-last_time)
                self.potential_mean         += self.potential_vector*(self.time-last_time)
                self.I_network[self.jump]   += 1
                self.total_jumps            += 1

        self.target_observable_mean = rate_diffs/(time_target-inner_time)
        
        # Final averages
        if self.total_jumps != 0:
            self.I_network          = self.I_network/self.total_jumps
        else:
            self.I_network          = np.zeros(len(self.adv_index_rows))
        self.charge_mean            = self.charge_mean/(time_target-inner_time)
        self.potential_mean         = self.potential_mean/(time_target-inner_time)
            
    def kmc_time_simulation_potential(self, target_electrode : int, time_target : float):
        """
        Runs KMC until KMC time exceeds a target value

        Parameters
        ----------
        target_electrode : int
            electrode index of which electric current is estimated
        time_target : float
            time value to be reached
        """

        # Calculate potential landscape once
        self.calc_potentials()
        
        self.total_jumps    = 0
        self.charge_mean    = np.zeros(len(self.charge_vector))
        self.potential_mean = np.zeros(len(self.potential_vector))
        self.I_network      = np.zeros(len(self.adv_index_rows))
        idx_np_target       = self.adv_index_cols[self.floating_electrodes]
        
        # Adjust floating target electrode once
        self.update_floating_electrode(idx_np_target)

        inner_time          = self.time
        last_time           = 0.0        
        target_potential    = 0.0

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

            # KMC Step and evolve in time
            self.select_event(random_number1, random_number2)

            if (self.jump == -1):

                # Update potential of floating target electrode
                self.update_floating_electrode(idx_np_target)

                # Update Observables
                target_potential    += self.potential_vector[target_electrode]*(time_target-last_time)
                self.charge_mean    += self.charge_vector*(time_target-last_time)
                self.potential_mean += self.potential_vector*(time_target-last_time)

                break

            # Occured jump
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time
            if (self.time >= time_target):
                self.neglect_last_event(np1,np2)

                # Update potential of floating target electrode
                self.update_floating_electrode(idx_np_target)

                # Update Observables
                target_potential    += self.potential_vector[target_electrode]*(time_target-last_time)
                self.charge_mean    += self.charge_vector*(time_target-last_time)
                self.potential_mean += self.potential_vector*(time_target-last_time)

                break

            else:

                # Update potential of floating target electrode
                self.update_floating_electrode(idx_np_target)

                # Update Observables
                target_potential            += self.potential_vector[target_electrode]*(self.time-last_time)
                self.charge_mean            += self.charge_vector*(self.time-last_time)
                self.potential_mean         += self.potential_vector*(self.time-last_time)
                self.I_network[self.jump]   += 1            
                self.total_jumps            += 1           
        
        self.target_observable_mean = target_potential/(time_target-inner_time)
        if self.total_jumps != 0:
            self.I_network          = self.I_network/self.total_jumps
        else:
            self.I_network          = np.zeros(len(self.adv_index_rows))
        self.charge_mean            = self.charge_mean/(time_target-inner_time)
        self.potential_mean         = self.potential_mean/(time_target-inner_time)
            
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
        
    def update_bimodal_resistance(self, slope : float, shift : float, R_max=30, R_min=20):

        r_max = R_max*self.ele_charge*self.ele_charge*1e-12
        r_min = R_min*self.ele_charge*self.ele_charge*1e-12

        self.resistances = (r_max - r_min)*(-np.tanh(slope*(self.I_tilde - shift)) + 1)/2 + r_min

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
      
    def return_target_values(self, output_potential=False):
        """
        Returns
        -------
        target_observable_mean : float
            Difference in target jumps towards/from target electrode
        target_observable_error : float
            Standard Deviation for difference in target jumps
        self.charge_mean/self.total_jumps : array
            Average charge landscape
        self.I_network
            Contribution of all tunnel junctions
        self.I_network_co
            Contribution of all cotunnel junctions
        self.total_jumps
            Number of total jumps
        """

        if output_potential:
            return self.target_observable_mean, self.target_observable_error, self.total_jumps

        else:
            return self.ele_charge*self.target_observable_mean*10**(-6), self.ele_charge*self.target_observable_error*10**(-6), self.total_jumps

    def return_average_charges(self):
        return self.charge_mean
    
    def return_average_potentials(self):
        return self.potential_mean
    
    def return_network_currents(self):
        return self.ele_charge*self.I_network*10**(-6)
    
    def return_average_resistances(self):
        return self.resistance_mean
    
    def return_landscape_per_it(self):
        return self.landscape_per_it
    
    def return_jump_dist_per_it(self):
        return self.jump_dist_per_it
    
    def return_time_vals(self):
        return self.time_values

###################################################################################################
###################################################################################################

def save_target_currents(output_values : List[np.array], voltages : np.array, path : str)->None:
    
    data    = np.hstack((voltages, output_values))
    columns = [f'E{i}' for i in range(voltages.shape[1]-1)]
    columns = np.array(columns + ['G', 'Eq_Jumps', 'Jumps', 'Current', 'Error'])

    df          = pd.DataFrame(data)
    df.columns  = columns

    df['Current']   = df['Current']
    df['Error']   = df['Error']

    if (os.path.isfile(path)):

        df.to_csv(path, mode='a', header=False, index=False)

    else:
        df.to_csv(path, header=True, index=False)

def save_mean_microstate(microstates : List[np.array], path : str)->None:

    # ele_charge     = 0.160217662
    # microstates_df = pd.DataFrame(microstates)/ele_charge
    microstates_df = pd.DataFrame(microstates)

    if (os.path.isfile(path)):
        microstates_df.to_csv(path, mode='a', header=False, index=False)
    else:
        microstates_df.to_csv(path, header=True, index=False)

def save_jump_storage(average_jumps : List[np.array], adv_index_rows : np.array, adv_index_cols : np.array, path : str)->None:

    avg_j_cols                  = [(adv_index_rows[i],adv_index_cols[i]) for i in range(len(adv_index_rows))]
    average_jumps_df            = pd.DataFrame(average_jumps)
    average_jumps_df.columns    = avg_j_cols
    average_jumps_df            = average_jumps_df

    if (os.path.isfile(path)):
        average_jumps_df.to_csv(path, mode='a', header=False, index=False)
    else:
        average_jumps_df.to_csv(path, header=True, index=False)

#####################################################################################################################################################################################
#####################################################################################################################################################################################

class simulation(tunneling.tunnel_class):

    def __init__(self, topology_parameter : dict, folder='', add_to_path="", res_info=None, res_info2=None,
                 np_info=None, np_info2=None, seed=None, high_C_output=False, **kwargs):
        """Defines network topology, electrostatic properties and tunneling junctions for a given type of topology. 

        Parameters
        ----------
        topology_parameter : dict
            Dictonary including information about number of nanoparticles, electrode positions and types.
        folder : str, optional
            Folder where simulation results are saved, by default ''
        add_to_path : str, optional
            String which is extended to the file path, by default ""
        res_info : dict, optional
            Dictonary including information about resistance values for the first type of nanoparticles, by default None
        res_info2 : dict, optional
            Dictonary including information about resistance values for the second type of nanoparticles, by default None
        np_info : dict, optional
            Dictonary including information about the first type of nanoparticles, by default None
        np_info2 : dict, optional
            Dictonary including information about the second type of nanoparticles, by default None
        seed : _type_, optional
            Seed in use, when network properties are randomly sampled, by default None
        high_C_output: bool, optional
            Add high capacitive output electrode, by default True

        Raises
        ------
        ValueError
            _description_
        """

        # Constant or floating electrodes
        electrode_type  = topology_parameter['electrode_type']

        # Inheritance 
        super().__init__(electrode_type, seed)

        # Type of Network Topology:
        if 'Nx' in topology_parameter:
            self.network_topology = 'cubic'
        else:
            self.network_topology = 'random'

        # Parameter of first nanoparticle type
        if np_info is None:
            np_info = {
                "eps_r"         : 2.6,  # Permittivity of molecular junction 
                "eps_s"         : 3.9,  # Permittivity of oxide layer
                "mean_radius"   : 10.0, # average nanoparticle radius
                "std_radius"    : 0.0,  # standard deviation of nanoparticle radius
                "np_distance"   : 1.0   # spacing between nanoparticle shells
            }

        # First type of nanoparticle resistances
        if res_info is None:
            res_info = {
                "mean_R"    : 25.0, # Average resistance
                "std_R"     : 0.0,  # Standard deviation of resistances
                "dynamic"   : False # Dynamic or constant resistances
            }
        
        # Resistance Information
        self.res_info   = res_info
        self.res_info2  = res_info2

        # For a cubic topology
        if self.network_topology == "cubic":

            # Path variable
            path_var = f'Nx={topology_parameter["Nx"]}_Ny={topology_parameter["Ny"]}_Nz={topology_parameter["Nz"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # Cubic Network Topology
            self.cubic_network(N_x=topology_parameter["Nx"], N_y=topology_parameter["Ny"], N_z=topology_parameter["Nz"])
            self.set_electrodes_based_on_pos(topology_parameter["e_pos"], topology_parameter["Nx"], topology_parameter["Ny"])
            if high_C_output:
                self.add_np_to_output()
            
            # Delete Junctions if porvided in kwargs
            if 'del_n_junctions' in kwargs:
                self.delete_n_junctions(kwargs['del_n_junctions'])
                
            # Electrostatic Properties
            self.init_nanoparticle_radius(np_info['mean_radius'], np_info['std_radius'])

            # Second Type of Nanopartciles
            if np_info2 != None:
                self.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'])

            # Capacitance Matrix
            self.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'], np_info['np_distance'])

        # For a disordered topology
        elif self.network_topology == "random":
            
            # Path variable
            path_var = f'Np={topology_parameter["Np"]}_Nj={topology_parameter["Nj"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # Random Network Topology
            self.random_network(N_particles=topology_parameter["Np"], N_junctions=topology_parameter["Nj"])
            self.add_electrodes_to_random_net(electrode_positions=topology_parameter["e_pos"])
            self.graph_to_net_topology()
                            
            # Electrostatic Properties
            self.init_nanoparticle_radius(np_info['mean_radius'], np_info['std_radius'])

            # Second Type of Nanopartciles
            if np_info2 != None:
                self.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'])

            # Capacitance Matrix
            self.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'], np_info['np_distance'])
        
        else:
            raise ValueError("Only 'cubic' and 'random' topologies are supported.")

        # Indices for Numpy broadcasting
        self.init_adv_indices()

        # Save Paths
        self.folder = folder
        self.path1  = folder + path_var
        self.path2  = folder + 'mean_state_'    + path_var
        self.path3  = folder + 'net_currents_'  + path_var

    def run_const_voltages(self, voltages : np.array, target_electrode : int, T_val=0.0, sim_dic=None, save_th=10, verbose=False):
        """Run a kinetic monte carlo simulation for constant electrode voltages to estimate either the constant electric current of
        the target electrode at constant target electrode voltage or the constant potential of a floating target electrode.

        Parameters
        ----------
        voltages : np.array
            2D-Array of electrode voltages with one simulation per row, and columns defining the electrode indices.
        target_electrode : int
            Index of electrode for which the electric current or potential is estimated 
        T_val : float, optional
            Network temperature, by default 0.0
        sim_dic : dict, optional
            Dictonary including simulation information, by default None
        save_th : int, optional
            Store simulation results after each save_th set electrode voltage combinations, by default 10
        output_potential : bool, optional
            If true, target electrode is floating and the simulation estimates its potential instead of the electric current, by default False
        verbose : bool, optional
            If true, track additional simulation observables. Increases simulation duration!
        """

        voltages    = np.round(voltages,5)
        
        # Simulation Parameter
        if sim_dic is None:
            sim_dic =   {
                "error_th"        : 0.05,
                "max_jumps"       : 10000000,
                "eq_steps"        : 100000,
                "jumps_per_batch" : 5000,
                "kmc_counting"    : False,
                "min_batches"     : 5
            }

        error_th        = sim_dic['error_th']           # Target obervable's relative error
        max_jumps       = sim_dic['max_jumps']          # Maximum number of KMC Steps
        eq_steps        = sim_dic['eq_steps']           # Equilibration Steps
        jumps_per_batch = sim_dic['jumps_per_batch']    # Number of Jumps per batch
        kmc_counting    = sim_dic['kmc_counting']       # Current calculation based on counting jumps
        min_batches     = sim_dic['min_batches']        # Minimum number of batches 

        # Electrode indices with floating or constant voltages
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        const_electrodes    = np.where(self.electrode_type == 'constant')[0]
        
        # Simulation Obseravles
        self.output_values              = []
        self.target_observable_values   = []
        self.microstates                = []
        self.landscape                  = []
        self.resistance_mean            = []
        self.pot_values                 = []
        self.jumps_per_it               = []
        self.time_values                = []
        self.pot_per_it                 = []
        self.average_jumps              = []
        self.average_cojumps            = []
        self.res_per_it                 = []

        # Check if target electrode is floating
        if self.electrode_type[target_electrode] == 'floating':
            output_potential=True
        else:
            output_potential=False

        j = 0
        
        # For each combination of electrode voltages
        for i, voltage_values in enumerate(voltages):
        
            # Based on current voltages get charges and potentials
            self.init_charge_vector(voltage_values=voltage_values)
            self.init_potential_vector(voltage_values=voltage_values)
            self.init_const_capacitance_values()

            # Return Model Arguments
            inv_capacitance_matrix          = self.return_inv_capacitance_matrix()
            charge_vector                   = self.return_charge_vector()
            potential_vector                = self.return_potential_vector()
            const_capacitance_values        = self.return_const_capacitance_values()
            N_electrodes, N_particles       = self.return_particle_electrode_count()
            adv_index_rows, adv_index_cols  = self.return_advanced_indices()
            temperatures                    = self.return_const_temperatures(T=T_val)
            resistances                     = self.return_random_resistances(R=self.res_info['mean_R'], Rstd=self.res_info['std_R'])
            resistances                     = self.ensure_undirected_resistances(resistances=resistances)

            # If second type of resistances is provided
            if self.res_info2 != None:
                resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

            # For memristive resistors
            if self.res_info['dynamic']:
                res_dynamic = self.res_info
            else:
                res_dynamic = False

            # Pass all model arguments into Numba optimized Class
            model = model_class(charge_vector, potential_vector, inv_capacitance_matrix,const_capacitance_values,
                                temperatures, resistances, adv_index_rows, adv_index_cols, N_electrodes, N_particles, floating_electrodes)

            if self.res_info['dynamic']:

                slope   = res_dynamic['slope']
                shift   = res_dynamic['shift']
                tau_0   = res_dynamic['tau_0']
                R_max   = res_dynamic['R_max']
                R_min   = res_dynamic['R_min']

                # Eqilibrate Potential Landscape
                eq_jumps = model.run_equilibration_steps_var_resistance(eq_steps, slope, shift, tau_0, R_max, R_min)
                
                # Production Run until Current at target electrode is less than error_th or max_jumps was passed
                model.kmc_simulation_var_resistance(target_electrode, error_th, max_jumps, jumps_per_batch, 
                                                    slope, shift, tau_0, R_max, R_min, kmc_counting, verbose)
            
            else:

                # Eqilibrate Potential Landscape
                eq_jumps = model.run_equilibration_steps(eq_steps)

                # Production Run until Current or Potential at target electrode is less than error_th or max_jumps was passed
                model.kmc_simulation(target_electrode, error_th, max_jumps, jumps_per_batch, output_potential, kmc_counting, min_batches, verbose)
            
            target_observable_mean, target_observable_error, total_jumps = model.return_target_values(output_potential)

            # Append Results to Outputs
            self.output_values.append(np.array([eq_jumps, total_jumps, target_observable_mean, target_observable_error]))
            self.microstates.append(model.return_average_charges())
            self.landscape.append(model.return_average_potentials())
            self.average_jumps.append(model.return_network_currents())

            if verbose:
                if output_potential:
                    self.target_observable_values.append(model.target_observable_values)
                else:
                    self.target_observable_values.append(self.ele_charge*model.target_observable_values*10**(-6))
                self.time_values.append(model.return_time_vals())
                self.pot_per_it.append(model.return_landscape_per_it())
                self.jumps_per_it.append(model.return_jump_dist_per_it())

                if self.res_info['dynamic']:
                    self.res_per_it.append(model.resistances_per_it)
                
            # Store Data
            if ((i+1) % save_th == 0):
                output_pots = self.landscape#[:,self.N_electrodes:]
                save_target_currents(np.array(self.output_values), voltages[j:(i+1),:], self.path1)
                save_mean_microstate(output_pots, self.path2)
                save_jump_storage(self.average_jumps, adv_index_rows, adv_index_cols, self.path3)
                self.output_values   = []
                self.microstates     = []
                self.landscape       = []
                self.average_jumps   = []
                self.average_cojumps = []
                j                    = i+1

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

        voltages    = np.round(voltages,5)

        # Check if target electrode is a floating electrode
        if self.electrode_type[target_electrode] == 'floating':
            output_potential    = True
        else:
            output_potential    = False
        
        # Electrode indices with floating or constant voltages
        floating_electrodes = np.where(self.electrode_type == 'floating')[0]
        const_electrodes    = np.where(self.electrode_type == 'constant')[0]

        # Init based on first time step
        self.init_charge_vector(voltage_values=voltages[0])
        self.init_potential_vector(voltage_values=voltages[0])
        self.init_const_capacitance_values()

        # Return Model Arguments
        inv_capacitance_matrix          = self.return_inv_capacitance_matrix()
        charge_vector                   = self.return_charge_vector()
        potential_vector                = self.return_potential_vector()
        const_capacitance_values        = self.return_const_capacitance_values()
        N_electrodes, N_particles       = self.return_particle_electrode_count()
        adv_index_rows, adv_index_cols  = self.return_advanced_indices()
        temperatures                    = self.return_const_temperatures(T=T_val)
        resistances                     = self.return_random_resistances(R=self.res_info['mean_R'], Rstd=self.res_info['std_R'])
        
        # Second resistor type
        if self.res_info2 is not None:
            resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

        # Pass all model arguments into Numba optimized Class
        self.model = model_class(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values,
                                temperatures, resistances, adv_index_rows, adv_index_cols,
                                N_electrodes, N_particles, floating_electrodes)

        # Resistors are Memristors
        if self.res_info['dynamic']:

            # Memristor Properties
            res_dynamic = self.res_info
            slope       = res_dynamic['slope']
            shift       = res_dynamic['shift']
            tau_0       = res_dynamic['tau_0']
            R_max       = res_dynamic['R_max']
            R_min       = res_dynamic['R_min']

            # Equilibration
            if init_charges is None:
                eq_jumps    = self.model.run_equilibration_steps_var_resistance(eq_steps, slope, shift, tau_0, R_max, R_min)
        else:

            res_dynamic = False

            # Equilibration
            if init_charges is None:
                eq_jumps    = self.model.run_equilibration_steps(eq_steps)

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
        self.output_values      = np.zeros(shape=(voltages.shape[0], 4))
        self.microstates        = np.zeros(shape=(voltages.shape[0], self.model.N_particles))
        self.landscape          = np.zeros(shape=(voltages.shape[0], self.model.N_particles+self.model.N_electrodes))
        self.average_jumps      = np.zeros(shape=(voltages.shape[0], len(self.model.adv_index_rows)))

        # Additional Observables
        if verbose:
            self.resistance_mean = np.zeros(shape=(voltages.shape[0], len(self.model.adv_index_rows)))
        
        # Store equilibrated charge distribution
        observable  = np.zeros(shape=(stat_size, len(voltages)))
        
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
                self.model.potential_vector[const_electrodes]  = voltage_values[const_electrodes]
                
                if self.res_info['dynamic']:
                    self.model.kmc_time_simulation_var_resistance(target_electrode, time_target, slope, shift, tau_0, R_max, R_min)
                else:
                    if output_potential:
                        self.model.kmc_time_simulation_potential(target_electrode, time_target)
                        target_observable_mean, target_observable_error, total_jumps = self.model.return_target_values(output_potential)

                    else:
                        self.model.kmc_time_simulation(target_electrode, time_target)
                        target_observable_mean, target_observable_error, total_jumps = self.model.return_target_values()
                
                # Add observables to outputs
                observable[s,i]         =  target_observable_mean
                self.output_values[i,:] += np.array([eq_jumps, total_jumps, 0.0, 0.0])/stat_size
                self.microstates[i,:]   += self.model.return_average_charges()/stat_size
                self.landscape[i,:]     += self.model.return_average_potentials()/stat_size
                self.average_jumps[i,:] += self.model.return_network_currents()/stat_size

                if verbose:
                    self.resistance_mean    += self.model.return_average_resistances()/stat_size
                
                # Subtract past charging state voltage contribution
                self.model.charge_vector    = self.model.charge_vector - offset

            # Last charge vector
            self.q_eq[s,:]   = self.model.charge_vector.copy()

        # Calculate Average observable and error
        self.output_values[:,2] = np.mean(observable, axis=0)
        self.output_values[:,3] = 1.96*np.std(observable, axis=0, ddof=1)/np.sqrt(stat_size)

        # Delte last row
        self.output_values      = np.delete(self.output_values,-1, axis=0)
        self.microstates        = np.delete(self.microstates,-1, axis=0)
        self.landscape          = np.delete(self.landscape,-1, axis=0)
        self.average_jumps      = np.delete(self.average_jumps,-1, axis=0)

        V_safe_vals                         = np.zeros(shape=(self.landscape.shape[0],self.N_electrodes+1))
        V_safe_vals[:,floating_electrodes]  = self.landscape[:,floating_electrodes]
        V_safe_vals[:,const_electrodes]     = voltages[:-1, const_electrodes]
        V_safe_vals[:,-1]                   = voltages[:-1, -1]

        if save:
            output_pots = self.landscape[:,self.N_electrodes:]
            save_target_currents(self.output_values, V_safe_vals, self.path1)
            save_mean_microstate(output_pots, self.path2)
            # save_mean_microstate(self.microstates@self.inv_capacitance_matrix, self.path2)
            save_jump_storage(self.average_jumps, adv_index_rows, adv_index_cols, self.path3)
        
    def clear_outputs(self):

        self.output_values   = []
        self.microstates     = []
        self.landscape       = []
        self.pot_values      = []
        self.jumps_per_it    = []
        self.time_values     = []
        self.pot_per_it      = []
        self.average_jumps   = []
        self.average_cojumps = []

    def return_output_values(self):

        return_var      = np.array(self.output_values)
        return_var[:,2] = return_var[:,2]
        return_var[:,3] = return_var[:,3]

        return return_var

    def return_microstates(self):

        return np.array(self.microstates)

    def return_potential_landscape(self):

        return np.array(self.landscape)

    def return_network_currents(self):

        avg_j_cols = [(self.adv_index_rows[i],self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))]

        return avg_j_cols, np.array(self.average_jumps)
    
    def return_jumps_per_it(self):

        return self.jumps_per_it
    
    def return_pot_per_it(self):

        return self.pot_per_it
    
    def return_time_vals(self):

        return self.time_values

###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':


    # Parameter
    N_x, N_y, N_z   = 3,1,1
    N_jumps         = 1000
    topology_string = {
        "Nx"                : N_x,
        "Ny"                : N_y,
        "Nz"                : N_z,
        "e_pos"             : [[0,0,0],[N_x-1,N_y-1,0]],
        "electrode_type"    : ['constant','floating']
    }
    topology        = {
        "Nx"                : N_x,
        "Ny"                : N_y,
        "Nz"                : N_z,
        "e_pos"             :  [[0,0,0],[int((N_x-1)/2),0,0],[N_x-1,0,0],[0,int((N_y-1)/2),0],[0,N_y-1,0],
                                [int((N_x-1)/2),N_y-1,0],[N_x-1,int((N_y-1)/2),0],[N_x-1,N_y-1,0]],
        "electrode_type"    : ['constant','floating','floating','floating','floating','floating','floating','floating']
    }
    sim_dic         = {
        "error_th"        : 0.0,      
        "max_jumps"       : N_jumps,
        "eq_steps"        : 0,
        "jumps_per_batch" : 1,
        "kmc_counting"    : False,
        "min_batches"     : 1
    }
    
    voltages_string     = np.array([[0.1,0.0,0.0]])
    voltages            = np.array([[0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    target_electrode    = len(topology_string['e_pos'])-1

    sim_class = simulation(network_topology='cubic', topology_parameter=topology_string)
    sim_class.run_const_voltages(voltages=voltages_string, target_electrode=target_electrode, save_th=0.1, output_potential=True, sim_dic=sim_dic)