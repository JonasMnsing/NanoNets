import multiprocessing
import tunneling
import numpy as np
import pandas as pd
import os.path
import copy
from typing import List
from numba.experimental import jitclass
# from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_type
from numba import int64, float64, boolean

spec = [
    ('charge_vector', float64[::1]),
    ('potential_vector', float64[::1]),
    ('tunnel_rates', float64[::1]),
    ('co_tunnel_rates', float64[::1]),
    ('inv_capacitance_matrix', float64[:,::1]),
    ('const_capacitance_values', float64[::1]),
    ('const_capacitance_values_co1', float64[::1]),
    ('const_capacitance_values_co2', float64[::1]),
    ('floating_electrodes', int64[::1]),
    ('temperatures', float64[::1]),
    ('temperatures_co', float64[::1]),
    ('resistances', float64[::1]),
    ('resistances_co1', float64[::1]),
    ('resistances_co2', float64[::1]),
    ('adv_index_rows', int64[::1]),
    ('adv_index_cols', int64[::1]),
    ('co_adv_index1', int64[::1]),
    ('co_adv_index2', int64[::1]),
    ('co_adv_index3', int64[::1]),
    ('N_electrodes', int64),
    ('N_particles', int64),
    ('ele_charge', float64),
    ('planck_const', float64),
    ('time', float64),
    ('counter_output_jumps_pos', float64),
    ('counter_output_jumps_neg', float64),
    ('target_observable_error_rel', float64),
    ('target_observable_mean', float64),
    ('target_observable_mean2', float64),
    ('target_observable_error', float64),
    ('target_observable_values', float64[::1]),
    ('time_values', float64[::1]),
    ('total_jumps', int64),
    ('jump', int64),
    ('cotunneling', boolean),
    ('co_event_selected', boolean),
    ('zero_T', boolean),
    ('charge_mean', float64[::1]),
    ('potential_mean', float64[::1]),
    ('resistance_mean', float64[::1]),
    ('I_network', float64[::1]),
    ('I_network_co', float64[::1]),
    ('I_tilde', float64[::1]),
    ('jump_dist_per_it', float64[:,::1]),
    ('resistances_per_it', float64[:,::1]),
    ('landscape_per_it', float64[:,::1]),
    ('time_vals', float64[::1]),
    ('N_rates', int64),
    ('N_corates', int64),
]

@jitclass(spec)
class model_class():
    """
    Numba optimized class to run the KMC procedure

    Attributes
    ----------
    N_particles : int
        Number of nanoparticles
    N_electrodes : int
        Number of electrodes
    N_rates : int
        Number of tunneling events
    N_corates : int
        Number of cotunneling events
    inv_capacitance_matrix : array
        Inverse of capacitance matrix
    cotunneling : bool
        Cotunneling or next neighbor tunneling
    co_event_selected : bool
        bool if cotunneling event was selected
    zero_T : bool
        bool if zero temerpature approximation for rates is true
    adv_index_rows : list
        Nanoparticles i (origin) in tunneling event i to j
    adv_index_cols : list
        Nanoparticles j (target) in tunneling event i to j
    co_adv_index1 : list
        Nanoparticles i (origin) in cotunneling event i to j via k
    co_adv_index2 : list
        Nanoparticles k in cotunneling event i to j via k
    co_adv_index3 : list
        Nanoparticles j (target) in cotunneling event i to j via k
    potential_vector : array
        Potential values for electrodes and nanoparticles
    charge_vector : array
        Charge values for each nanoparticle 
    tunnel_rates : array
        Tunnel rate values
    co_tunnel_rates : array
        Cotunnel rate values
    const_capacitance_values : array
        Sum of capacitance for free energy calculation
    const_capacitance_values_co1 : array
        Sum of capacitance for free energy calculation in cotunneling
    const_capacitance_values_co2 : array
        Sum of capacitance for free energy calculation in cotunneling
    temperatures : array
        temperature values for each tunnel event
    temperatures_co : array
        temperature values for each cotunnel event
    resistances : array
        Resistances for each tunneling event i to j
    resistances_co1 : array
        Resistances for each cotunneling event i to k
    resistances_co2 : array
        Resistances for each cotunneling event k to j
    counter_output_jumps_pos : int
        Number of Jumps towards target electrode
    counter_output_jumps_neg : int
        Number of Jumps from target electrode
    total_jumps : int
        Total number of Jumps / KMC Steps
    time : float
        KMC time scale
    target_observable_mean : float
        Difference in Jumps towards/from target electrode
    target_observable_mean2 : float
        Difference in Jumps towards/from target electrode
    target_observable_error : float
        Standard Deviation for difference in jumps
    rel_error : float
        Relative Error for difference in jumps
    jump : int
        Occured jump/event
    charge_mean : array
        Storage for average network charges
    potential_mean : array
        Storage for average potential landscape
    jump_storage : array
        Storage for executed tunneling events
    jump_storage_co : array
        Storage for executed cotunneling events

    Methods
    -------
    calc_potentials()
        Compute potentials via matrix vector multiplication
    update_potentials(np1, np2)
        Update potentials after occured jump
    calc_tunnel_rates()
        Compute tunnel rates
    calc_tunnel_rates_zero_T()
        Compute tunnel rates in zero T approximation
    calc_cotunnel_rates()
        Compute cotunnel rates
    calc_cotunnel_rates_zero_T()
        Compute cotunnel rates in zero T approximation
    calc_rel_error()
        Calculate relative error and standard deviation by welford one pass
    select_event(random_number1 : float, random_number2 : float)
        Select next charge hopping event and update time by kinetic monte carlo apporach.
        Updates given charge vector based on executed event
    select_co_event(random_number1 : float, random_number2 : float)
        Select next charge hopping event and update time by kinetic monte carlo apporach considering cotunneling
        Updates given charge vector based on executed event
        NEEDS UPDATE TO NEW POTENTIAL CALCULATION. DOES NOT WORK!!!
    reach_equilibrium(min_jumps=1000, max_steps=10, rel_error=0.15, min_nps_eq=0.9, max_jumps=1000000)
        Equilibrate the system
    kmc_simulation(target_electrode : int, error_th = 0.05, max_jumps=10000000)
        Runs KMC until current for target electrode has a relative error below error_th or max_jumps is reached
        Tracks Mean Current, Current standard deviation, average microstate (charge vector), contribution of each junction
    kmc_time_simulation
        Runs KMC until KMC time exceeds a target value
    return_target_values()
    """

    def __init__(self, charge_vector, potential_vector, inv_capacitance_matrix,
                const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2,
                temperatures, temperatures_co, resistances, resistances_co1,
                resistances_co2, adv_index_rows, adv_index_cols, co_adv_index1,
                co_adv_index2, co_adv_index3, N_electrodes, N_particles, floating_electrodes):

        # np.random.seed(42)

        # CONST
        self.ele_charge     = 0.160217662
        self.planck_const   = 1.054571817*10**(-16)

        # Previous class attributes
        self.charge_vector                  = charge_vector
        self.potential_vector               = potential_vector
        self.inv_capacitance_matrix         = inv_capacitance_matrix
        self.const_capacitance_values       = const_capacitance_values
        self.const_capacitance_values_co1   = const_capacitance_values_co1
        self.const_capacitance_values_co2   = const_capacitance_values_co2
        self.floating_electrodes            = floating_electrodes
        self.temperatures                   = temperatures
        self.temperatures_co                = temperatures_co
        self.resistances                    = resistances
        self.resistances_co1                = resistances_co1
        self.resistances_co2                = resistances_co2
        self.adv_index_rows                 = adv_index_rows
        self.adv_index_cols                 = adv_index_cols
        self.co_adv_index1                  = co_adv_index1
        self.co_adv_index2                  = co_adv_index2
        self.co_adv_index3                  = co_adv_index3
        self.N_electrodes                   = N_electrodes
        self.N_particles                    = N_particles
        self.N_rates                        = len(self.adv_index_rows)
        self.N_corates                      = len(self.co_adv_index1)

        # Simulation attributes
        self.counter_output_jumps_pos       = 0      
        self.counter_output_jumps_neg       = 0      
        self.total_jumps                    = 0      
        self.time                           = 0.0    
        self.target_observable_mean         = 0.0    
        self.target_observable_mean2        = 0.0   
        self.target_observable_error        = 0.0
        self.target_observable_error_rel    = 1.0 
        self.jump                           = 0

        # Storages
        self.charge_mean        = np.zeros(len(charge_vector))
        self.potential_mean     = np.zeros(len(potential_vector))
        self.resistance_mean    = np.zeros(len(resistances))
        self.I_network          = np.zeros(len(adv_index_rows))
        self.I_network_co       = np.zeros(len(co_adv_index1))
        self.I_tilde            = np.zeros(len(self.adv_index_rows))

        # Co-tunneling bools
        self.cotunneling        = False
        self.co_event_selected  = False
        self.zero_T             = False
        
        if (np.sum(self.temperatures) == 0.0):
            self.zero_T = True        

        if (len(self.co_adv_index1) != 0):
            self.cotunneling = True
        
    def calc_potentials(self):
        """Compute potentials via matrix vector multiplication
        """

        self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)

    def update_potentials(self, np1, np2):
        """Update potentials after occured jump

        Parameters
        ----------
        np1 : int
            Last jump's origin
        np2 : int
            Last jump's target
        """

        if ((np1 - self.N_electrodes) < 0):
            self.potential_vector[self.N_electrodes:] = self.potential_vector[self.N_electrodes:] + self.inv_capacitance_matrix[:,np2]*self.ele_charge
        elif ((np2 - self.N_electrodes) < 0):
            self.potential_vector[self.N_electrodes:] = self.potential_vector[self.N_electrodes:] - self.inv_capacitance_matrix[:,np1]*self.ele_charge
        elif ((np1 - self.N_electrodes) < 0 and ((np2 - self.N_electrodes) < 0)):
            self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)
        else:
            self.potential_vector[self.N_electrodes:] = self.potential_vector[self.N_electrodes:] + (self.inv_capacitance_matrix[:,np2]
                                                        - self.inv_capacitance_matrix[:,np1])*self.ele_charge
    
    def update_floating_electrode(self, idx_np_target)->None:
        """Update potentials of floating electrodes based on capacitance. Update charge vector given new potentials.

        Parameters
        ----------
        idx_np_target : np.array
            Indices of nanoparticles adjacent to electrodes  
        """

        self.potential_vector[self.floating_electrodes] = self.potential_vector[idx_np_target]
        
    def calc_tunnel_rates(self):
        """Compute tunneling rates
        """

        free_energy         = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        self.tunnel_rates   = (free_energy/self.resistances)/(np.exp(free_energy/self.temperatures) - 1.0)

    def calc_tunnel_rates_zero_T(self):
        """Compute tunneling rates in zero T approximation
        """
                
        free_energy = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        
        self.tunnel_rates                   = np.zeros(self.N_rates)
        self.tunnel_rates[free_energy<0]    = -free_energy[free_energy<0]/self.resistances[free_energy<0]

    def calc_cotunnel_rates(self):
        """Compute cotunneling rates
        """

        free_energy1    = self.ele_charge*(self.potential_vector[self.co_adv_index1] - self.potential_vector[self.co_adv_index2]) + self.const_capacitance_values_co1
        free_energy2    = self.ele_charge*(self.potential_vector[self.co_adv_index2] - self.potential_vector[self.co_adv_index3]) + self.const_capacitance_values_co2

        factor          = self.planck_const/(3*np.pi*(self.ele_charge**4)*self.resistances_co1*self.resistances_co2)
        val1            = free_energy2/(4*free_energy1**2 - 4*free_energy1*free_energy2 + free_energy2**2)
        val2            = ((2*np.pi*self.temperatures_co)**2 + free_energy2**2)/(np.exp(free_energy2/self.temperatures_co) - 1.0)
        
        self.co_tunnel_rates = factor*val1*val2
    
    def calc_cotunnel_rates_zero_T(self):
        """Compute cotunneling rates in zero T approximation
        """

        free_energy1    = self.ele_charge*(self.potential_vector[self.co_adv_index1] - self.potential_vector[self.co_adv_index2]) + self.const_capacitance_values_co1
        free_energy2    = self.ele_charge*(self.potential_vector[self.co_adv_index2] - self.potential_vector[self.co_adv_index3]) + self.const_capacitance_values_co2

        factor                  = self.planck_const/(3*np.pi*(self.ele_charge**4)*self.resistances_co1*self.resistances_co2)
        val1                    = free_energy2/(4*free_energy1**2 - 4*free_energy1*free_energy2 + free_energy2**2)
        val2                    = np.zeros(self.N_corates)
        val2[free_energy2<0]    = -free_energy2[free_energy2<0]**2

        self.co_tunnel_rates = factor*val1*val2
    
    def calc_rel_error(self, N_calculations):
        """Calculate relative error and standard deviation of target observable via welford one pass 
        """

        if N_calculations >= 2:

            self.target_observable_error     = 1.96*np.sqrt(np.abs(self.target_observable_mean2) / (N_calculations - 1))/np.sqrt(N_calculations)
            self.target_observable_error_rel = self.target_observable_error/np.abs(self.target_observable_mean)
    
    def select_event(self, random_number1 : float, random_number2 : float):
        """Select next charge hopping event and update time by kinetic monte carlo apporach.
        
        Parameters
        ----------
        random_number1 : float
            First random number
        random_number2 : float
            Second random number
        """

        # Get CUMSUM of Tunnel Rates
        kmc_cum_sum = np.cumsum(self.tunnel_rates)
        k_tot       = kmc_cum_sum[-1]
        event       = random_number1 * k_tot

        if (k_tot == 0.0):
            # self.time   = 1.0
            self.jump   = -1
            return
        
        # Select next Jump
        jump    = np.searchsorted(a=kmc_cum_sum, v=event)   
        np1     = self.adv_index_rows[jump]
        np2     = self.adv_index_cols[jump]

        # If Electrode is involved
        if ((np1 - self.N_electrodes) < 0):
            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
            self.potential_vector[self.N_electrodes:] += self.ele_charge*self.inv_capacitance_matrix[:,np2-self.N_electrodes]
        elif ((np2 - self.N_electrodes) < 0):
            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
            self.potential_vector[self.N_electrodes:] -= self.ele_charge*self.inv_capacitance_matrix[:,np1-self.N_electrodes]
        else:
            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
            self.potential_vector[self.N_electrodes:] += self.ele_charge*(self.inv_capacitance_matrix[:,np2-self.N_electrodes] - self.inv_capacitance_matrix[:,np1-self.N_electrodes])
        
        # Update Time and last Jump
        self.time   = self.time - np.log(random_number2)/k_tot
        self.jump   = jump

    def neglect_last_event(self, np1, np2):
        """Reverse last tunneling event

        Parameters
        ----------
        np1 : int
            Nanoparticle index 1
        np2 : int
            Nanoparticle index 2
        """
        # If Electrode is involved
        if ((np1 - self.N_electrodes) < 0):
            self.charge_vector[np2-self.N_electrodes] -= self.ele_charge
            self.potential_vector[self.N_electrodes:] -= self.ele_charge*self.inv_capacitance_matrix[:,np2-self.N_electrodes]
        elif ((np2 - self.N_electrodes) < 0):
            self.charge_vector[np1-self.N_electrodes] += self.ele_charge
            self.potential_vector[self.N_electrodes:] += self.ele_charge*self.inv_capacitance_matrix[:,np1-self.N_electrodes]
        else:
            self.charge_vector[np1-self.N_electrodes] += self.ele_charge
            self.charge_vector[np2-self.N_electrodes] -= self.ele_charge
            self.potential_vector[self.N_electrodes:] -= self.ele_charge*(self.inv_capacitance_matrix[:,np2-self.N_electrodes] - self.inv_capacitance_matrix[:,np1-self.N_electrodes])
    
    # TODO Does not work!
    def select_co_event(self, random_number1 : float, random_number2 : float):
        """Select next charge hopping event and update time by kinetic monte carlo apporach considering cotunneling.
        NEEDS UPDATE TO NEW POTENTIAL CALCULATION. DOES NOT WORK!!!

        Parameters
        ----------
        random_number1 : float
            First random number
        random_number2 : float
            Second random number
        """

        sum_a       = np.sum(self.tunnel_rates)
        sum_b       = np.sum(self.co_tunnel_rates)
        k_tot       = sum_a + sum_b
        event       = random_number1 * k_tot
        min_val     = 0.0
        max_val     = 0.0

        self.co_tunnel_rates += sum_a

        if (k_tot == 0.0):
            
            self.time   = 1.0
            self.jump   = -1
            return
        
        if event <= sum_a:

            self.co_event_selected = False
            
            for jump, t_rate in enumerate(self.tunnel_rates):
                if (t_rate != 0.0):

                    max_val = min_val + t_rate

                    if ((event > min_val) and (event <= max_val)):

                        np1 = self.adv_index_rows[jump]
                        np2 = self.adv_index_cols[jump]

                        if ((np1 - self.N_electrodes) < 0):
                            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
                        elif ((np2 - self.N_electrodes) < 0):
                            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
                        else:
                            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
                            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
                        
                        break

                    min_val = max_val
        
        else:
            
            self.co_event_selected  = True
            min_val                 = sum_a

            for jump, t_rate in enumerate(self.co_tunnel_rates):

                if (t_rate != sum_a):

                    max_val = min_val + t_rate

                    if ((event > min_val) and (event <= max_val)):

                        np1 = self.co_adv_index1[jump]
                        np2 = self.co_adv_index3[jump]

                        if ((np1 - self.N_electrodes) < 0):
                            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
                        elif ((np2 - self.N_electrodes) < 0):
                            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
                        else:
                            self.charge_vector[np1-self.N_electrodes] -= self.ele_charge
                            self.charge_vector[np2-self.N_electrodes] += self.ele_charge
                        
                        break

                    min_val = max_val
        
        self.time   = self.time - np.log(random_number2)/k_tot
        self.jump   = jump
    
    def run_equilibration_steps(self, n_jumps=10000):
        """Execute a fixed amount of KMC steps for equilibration.

        Parameters
        ----------
        n_jumps : int, optional
            Number of executed KMC steps, by default 10000

        Returns
        -------
        int
            Number of executed KMC steps
        """
        
        idx_np_target = self.adv_index_cols[self.floating_electrodes]

        self.calc_potentials()
        self.update_floating_electrode(idx_np_target)

        for i in range(n_jumps):

            if (self.jump == -1):
                break
            
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            if self.cotunneling == False:
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()
                self.select_event(random_number1, random_number2)
            else:
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                    self.calc_cotunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()
                    self.calc_cotunnel_rates_zero_T()
                self.select_co_event(random_number1, random_number2)
            
            self.update_floating_electrode(idx_np_target)
            
        return n_jumps
    
    def run_equilibration_steps_var_resistance(self, n_jumps=10000, slope=0.8, shift=7.5,
                                               tau_0=1e-8, R_max=25, R_min=10):
        """Execute a fixed amount of KMC steps for equilibration considering Memristors.

        Parameters
        ----------
        n_jumps : int, optional
            Number of executed KMC steps, by default 10000
        slope : float, optional
            Memristor slope, by default 0.8
        shift : float, optional
            Memristor shifft, by default 7.5
        tau_0 : _type_, optional
            Memory constant, by default 1e-8
        R_max : int, optional
            Maximum Resistance, by default 25
        R_min : int, optional
            Minimum Resistance, by default 10

        Returns
        -------
        int
            Number of executed KMC steps
        """

        self.I_tilde = np.zeros(len(self.adv_index_rows))

        self.calc_potentials()

        for i in range(n_jumps):

            self.update_bimodal_resistance(slope, shift, R_max, R_min)
            
            t1 = self.time

            if (self.jump == -1):
                break
            
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()

            if self.cotunneling == False:
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()
                self.select_event(random_number1, random_number2)
            else:
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                    self.calc_cotunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()
                    self.calc_cotunnel_rates_zero_T()
                self.select_co_event(random_number1, random_number2)
            
            t2  = self.time
            dt  = t2-t1

            self.I_tilde             = self.I_tilde*np.exp(-dt/tau_0)
            self.I_tilde[self.jump]  += 1


        return n_jumps
     
    def return_next_means(self, new_value, mean_value, mean_value2, count):

        count           +=  1
        delta           =   new_value - mean_value
        mean_value      +=  delta/count          
        delta2          =   new_value - mean_value
        mean_value2     +=  delta * delta2

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

        self.total_jumps                    = 0     # Total number of KMC Steps
        self.target_observable_error_rel    = 1.0   # Observable's relative error
        self.target_observable_mean         = 0.0   # Average oberservable values
        self.target_observable_mean2        = 0.0   # Helper value
        self.target_observable_error        = 0.0   # Observable's standard error

        idx_np_target                       = self.adv_index_cols[self.floating_electrodes]

        self.charge_mean        = np.zeros(len(self.charge_vector))     # Average charge distribution
        self.potential_mean     = np.zeros(len(self.potential_vector))  # Average potential landscape
        self.I_network          = np.zeros(len(self.adv_index_rows))    # Network electric currents

        # If current based on tunnel rates, return target rate indices
        if kmc_counting == False:
            rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
            rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]

        # For additional information, track batched values
        if verbose:
            self.target_observable_values   = np.zeros(int(max_jumps/jumps_per_batch))                                  # Observable in each batch
            self.landscape_per_it           = np.zeros((int(max_jumps/jumps_per_batch), len(self.potential_vector)))    # Potential landscape in each batch
            self.time_values                = np.zeros(int(max_jumps/jumps_per_batch))                                  # Time passed during each batch
            self.jump_dist_per_it           = np.zeros((int(max_jumps/jumps_per_batch), len(self.adv_index_rows)))      # Jumps occured during each batch for each junction

        count       = 0     # Number of while loops
        below_rel   = 0     # Number of consecutive sample below relative error
        time_total  = 0.0   # Total time passed

        # Calculate potential landscape once
        self.calc_potentials()

        # Calculate floating electrode voltages once
        self.update_floating_electrode(idx_np_target)

        # While relative error or maximum amount of KMC steps not reached
        while((below_rel < min_batches) and (self.total_jumps < max_jumps)):
            
            # Counting or not
            if kmc_counting:
                self.counter_output_jumps_pos   = 0
                self.counter_output_jumps_neg   = 0
            else:
                target_value                    = 0.0

            # Reset Arrays for given batch
            jump_storage_vals   = np.zeros(len(self.adv_index_rows))
            time_values         = np.zeros(jumps_per_batch)
            charge_values       = np.zeros(self.N_particles)
            potential_values    = np.zeros(self.N_particles+self.N_electrodes)
            self.time           = 0.0

            # KMC Run for a batch
            for i in range(jumps_per_batch):
                
                # Start time
                t1 = self.time
            
                # KMC Part
                random_number1  = np.random.rand()
                random_number2  = np.random.rand()

                # T=0 Approximation of Rates
                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()

                # Without counting, extract rate difference
                if kmc_counting == False:
                    # rate_diffs[i] = self.tunnel_rates[rate_index1] - self.tunnel_rates[rate_index2]
                    rate1   = self.tunnel_rates[rate_index1]
                    rate2   = self.tunnel_rates[rate_index2]
                
                # KMC Step and evolve in time
                self.select_event(random_number1, random_number2)

                # Update potentials of floating electrodes
                self.update_floating_electrode(idx_np_target)

                if (self.jump == -1):
                    break

                # Occured jump
                np1 = self.adv_index_rows[self.jump]
                np2 = self.adv_index_cols[self.jump]
                jump_storage_vals[self.jump]    += 1

                # End time and difference
                t2              = self.time
                time_values[i]  = t2-t1

                # Add charge and potential vector
                charge_values       += self.charge_vector*(t2-t1)
                potential_values    += self.potential_vector*(t2-t1)

                if output_potential:
                    target_value += self.potential_vector[target_electrode]*(t2-t1)
                else:
                    if kmc_counting:
                        # If jump from target electrode
                        if (np1 == target_electrode):
                            self.counter_output_jumps_neg += 1
                            
                        # If jump towards target electrode
                        if (np2 == target_electrode):
                            self.counter_output_jumps_pos += 1
                    else:
                        target_value += (rate1 - rate2)*(t2-t1)

            # Update total jumps, average charges, and average potentials
            self.total_jumps    += i+1
            time_total          += self.time

            if self.time != 0:

                self.charge_mean    += charge_values
                self.potential_mean += potential_values

                # Calc new average currents
                if kmc_counting and not(output_potential):
                    target_observable   = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
                else:
                    target_observable   = target_value/self.time

                if verbose:
                    self.target_observable_values[count]    = target_observable
                    self.time_values[count]                 = self.time
                    self.landscape_per_it[count,:]          = potential_values/self.time
                    self.jump_dist_per_it[count,:]          = jump_storage_vals
                
                self.target_observable_mean, self.target_observable_mean2, count    = self.return_next_means(target_observable, self.target_observable_mean, self.target_observable_mean2, count)
                self.I_network                                                      += jump_storage_vals/self.time

                if (self.target_observable_mean != 0):
                    self.calc_rel_error(count)

                if self.target_observable_error_rel < error_th:
                    below_rel += 1
                else:
                    below_rel = 0

            else:
                self.charge_mean    += self.charge_vector
                self.potential_mean += self.potential_vector
                self.I_network      = jump_storage_vals/self.total_jumps

                if not(output_potential):
                    self.target_observable_mean = 0.0
                else:
                    self.target_observable_mean = self.potential_vector[target_electrode]
            
            if (self.jump == -1):
                break



            # if (self.jump == -1):

            #     self.charge_mean    += charge_values
            #     self.potential_mean += potential_values
            #     self.I_network      = jump_storage_vals/self.total_jumps
            #     time_total          += self.time

            #     if self.time != 0:
            #         target_observable   = target_value/self.time
            #     else:
            #         target_observable   = self.potential_vector[target_electrode]

            #     # self.charge_mean    = self.charge_vector
            #     # self.potential_mean = self.potential_vector
            #     # self.I_network      = jump_storage_vals/self.total_jumps

            #     # if not(output_potential):
            #     #     self.target_observable_mean = 0.0
            #     # else:
            #     #     self.target_observable_mean = self.potential_vector[target_electrode]
            #     break

            # else:
            #     self.charge_mean    += charge_values
            #     self.potential_mean += potential_values
            #     time_total          += self.time

            #     if self.time != 0:
                    
            #         # Calc new average currents
            #         if kmc_counting and not(output_potential):
            #             target_observable   = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
            #         else:
            #             target_observable   = target_value/self.time

            #         if verbose:
            #             self.target_observable_values[count]    = target_observable
            #             self.time_values[count]                 = self.time
            #             self.landscape_per_it[count,:]          = potential_values/self.time
            #             self.jump_dist_per_it[count,:]          = jump_storage_vals

            #         self.target_observable_mean, self.target_observable_mean2, count    = self.return_next_means(target_observable, self.target_observable_mean, self.target_observable_mean2, count)
            #         self.I_network                                                      += jump_storage_vals/self.time

            # # if (self.jump == -1):
            # #     if not(output_potential):
            # #         self.target_observable_mean = 0.0
            # #     else:
            # #         self.target_observable_mean = self.potential_vector[target_electrode]
            # #     break

            #         if (self.target_observable_mean != 0):
            #             self.calc_rel_error(count)

            #         if self.target_observable_error_rel < error_th:
            #             below_rel += 1
            #         else:
            #             below_rel = 0

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

        # Calculate potential landscape once
        self.calc_potentials()

        self.total_jumps    = 0
        self.charge_mean    = np.zeros(len(self.charge_vector))
        self.potential_mean = np.zeros(len(self.potential_vector))
        self.I_network      = np.zeros(len(self.adv_index_rows))

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
    
                # Update Observables
                rate_diffs                  += (rate1 - rate2)*(time_target-last_time)
                self.charge_mean            += self.charge_vector*(time_target-last_time)
                self.potential_mean         += self.potential_vector*(time_target-last_time)

                break

            # Occured jump
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time
            if (self.time >= time_target):

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
    
    def kmc_time_simulation_potential2(self, target_electrode : int, time_target : float):
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

        inner_time          = self.time
        last_time           = 0.0        
        target_potential    = 0.0

        while (self.time < time_target):

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
                break

            # Occured jump
            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            # If time exceeds target time
            if (self.time >= time_target):
                self.neglect_last_event(np1,np2)
                break

            # Update potential of floating target electrode
            self.update_floating_electrode(idx_np_target)
            target_potential += self.potential_vector[target_electrode]*(self.time-last_time)

            # Update Observables
            self.charge_mean            += self.charge_vector*(self.time-last_time)
            self.potential_mean         += self.potential_vector*(self.time-last_time)
            self.I_network[self.jump]   += 1            
            self.total_jumps            += 1           
        
        if (self.jump == -1):
            self.target_observable_mean  = self.potential_vector[target_electrode]

        if (last_time-inner_time) != 0:
            
            self.target_observable_mean = target_potential/(time_target-inner_time)
            self.I_network              = self.I_network/self.total_jumps
            self.charge_mean            = self.charge_mean/(time_target-inner_time)
            self.potential_mean         = self.potential_mean/(time_target-inner_time)
            
        else:
            self.target_observable_mean = self.potential_vector[target_electrode]
            self.I_network              = self.I_network
            self.charge_mean            = self.charge_vector
            self.potential_mean         = self.potential_vector
        
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
# FUNCTIONS
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

def save_cojump_storage(average_cojumps : List[np.array], co_adv_index1 : np.array, co_adv_index3 : np.array, path : str)->None:

    avg_j_cols                  = [(co_adv_index1[i],co_adv_index3[i]) for i in range(len(co_adv_index1))]
    average_jumps_df            = pd.DataFrame(average_cojumps)
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
                 np_info=None, np_info2=None, seed=None, **kwargs):
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

        Raises
        ------
        ValueError
            _description_
        """

        # Constant or floating electrodes
        electrode_type  = topology_parameter['electrode_type']

        # Inheritance 
        super().__init__(electrode_type, 1, seed)

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
            self.add_np_to_e_pos()
            
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
        self.path2  = folder + 'mean_state_' + path_var
        self.path3  = folder + 'net_currents_' + path_var

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
            inv_capacitance_matrix                                                                  = self.return_inv_capacitance_matrix()
            charge_vector                                                                           = self.return_charge_vector()
            potential_vector                                                                        = self.return_potential_vector()
            const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2    = self.return_const_capacitance_values()
            N_electrodes, N_particles                                                               = self.return_particle_electrode_count()
            adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2, co_adv_index3             = self.return_advanced_indices()
            temperatures, temperatures_co                                                           = self.return_const_temperatures(T=T_val)
            resistances, resistances_co1, resistances_co2                                           = self.return_random_resistances(R=self.res_info['mean_R'], Rstd=self.res_info['std_R'])
            resistances                                                                             = self.ensure_undirected_resistances(resistances=resistances)

            # If second type of resistances is provided
            if self.res_info2 != None:
                resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

            # For memristive resistors
            if self.res_info['dynamic']:
                res_dynamic = self.res_info
            else:
                res_dynamic = False

            # Pass all model arguments into Numba optimized Class
            model = model_class(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2,
                                temperatures, temperatures_co, resistances, resistances_co1, resistances_co2, adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2,
                                co_adv_index3, N_electrodes, N_particles, floating_electrodes)

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
                
                save_target_currents(np.array(self.output_values), voltages[j:(i+1),:], self.path1)
                save_mean_microstate(self.microstates, self.path2)
                save_jump_storage(self.average_jumps, adv_index_rows, adv_index_cols, self.path3)
                if (self.tunnel_order != 1):
                    save_cojump_storage(self.average_cojumps, co_adv_index1, co_adv_index3, self.path4)
                self.output_values   = []
                self.microstates     = []
                self.landscape       = []
                self.average_jumps   = []
                self.average_cojumps = []
                j                    = i+1

    def run_var_voltages(self, voltages : np.array, time_steps : np.array, target_electrode : int, T_val=0.0, eq_steps=0, save=True,
                         stat_size=500, init_charges=None, verbose=False):
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
        inv_capacitance_matrix                                                                  = self.return_inv_capacitance_matrix()
        charge_vector                                                                           = self.return_charge_vector()
        potential_vector                                                                        = self.return_potential_vector()
        const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2    = self.return_const_capacitance_values()
        N_electrodes, N_particles                                                               = self.return_particle_electrode_count()
        adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2, co_adv_index3             = self.return_advanced_indices()
        temperatures, temperatures_co                                                           = self.return_const_temperatures(T=T_val)
        resistances, resistances_co1, resistances_co2                                           = self.return_random_resistances(R=self.res_info['mean_R'], Rstd=self.res_info['std_R'])
        
        # Second resistor type
        if self.res_info2 is not None:
            resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

        # Pass all model arguments into Numba optimized Class
        self.model = model_class(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values, const_capacitance_values_co1,const_capacitance_values_co2,
                                temperatures, temperatures_co, resistances, resistances_co1, resistances_co2, adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2,
                                co_adv_index3, N_electrodes, N_particles, floating_electrodes)

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
        observable          = np.zeros(shape=(stat_size, len(voltages)))
        
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
                    self.resistance_mean += self.model.return_average_resistances()/stat_size
                
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
        
    def train_time_series(self, x : np.array, y : np.array, learning_rate : float, batch_size : int, N_epochs : int, epsilon=0.001, adam=False,
                          time_step=1e-10, stat_size=500, Uc_init=0.1, transient_steps=0, print_nth_batch=1, save_nth_epoch=1, path=''):

        # Parameter
        N_voltages          = len(x)
        N_batches           = int(N_voltages/batch_size)
        N_controls          = self.N_electrodes - 2
        time_steps          = time_step*np.arange(N_voltages)
        control_voltages    = np.random.uniform(-Uc_init, Uc_init, N_controls)
        target_electrode    = self.N_electrodes - 1
        y                   = (y - np.mean(y))/np.std(y)

        # Multiprocessing Manager
        with multiprocessing.Manager() as manager:

            # Storage for simulation results
            return_dic = manager.dict()
            current_charge_vector = None
            
            # ADAM Optimization
            if adam:
                m = np.zeros_like(control_voltages)
                v = np.zeros_like(control_voltages)

            for epoch in range(1, N_epochs+1):
                
                # Set charge vector to None
                predictions = np.zeros(N_voltages)
                
                for batch in range(N_batches):

                    start   = batch*batch_size
                    stop    = (batch+1)*batch_size
                    
                    # Voltage array containing input at column 0 
                    voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
                    voltages[:,0]       = x[start:stop]
                    voltages[:,1:-2]    = np.tile(np.round(control_voltages,4), (batch_size,1))
                    voltages_list       = []
                    voltages_list.append(voltages)

                    # Set up a list of voltages considering small deviations
                    for i in range(N_controls):

                        voltages_tmp        = voltages.copy()
                        voltages_tmp[:,i+1] += epsilon
                        voltages_list.append(voltages_tmp)

                        voltages_tmp        = voltages.copy()
                        voltages_tmp[:,i+1] -= epsilon
                        voltages_list.append(voltages_tmp)

                    # Container for processes
                    procs = []

                    # For each set of voltages assign start a process
                    for thread in range(2*N_controls+1):

                        # Make a deep copy of the current instance for each process
                        instance_copy   = copy.deepcopy(self)

                        # Start process
                        process = multiprocessing.Process(target=self.sim_run_for_gradient_decent, args=(thread, instance_copy, return_dic, voltages_list[thread],
                                                                                                         time_steps[start:stop], target_electrode, stat_size,
                                                                                                         current_charge_vector))
                        process.start()
                        procs.append(process)
                    
                    # Wait for all processes
                    for p in procs:
                        p.join()

                    # Current charge vector given the last iteration
                    current_charge_vector = return_dic[-1]
                    
                    # Gradient Container
                    gradients = np.zeros_like(control_voltages)

                    # Calculate gradients for each control voltage 
                    for i in np.arange(1,2*N_controls+1,2):

                        y_pred_eps_pos          = return_dic[i]
                        y_pred_eps_neg          = return_dic[i+1]
                        y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
                        y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
                        perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
                        perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
                        gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                    
                    if ((epoch != 1) and (batch != 0)):
                        
                        # Current prediction and loss
                        y_pred  = return_dic[0]
                        y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
                        loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                        
                        predictions[(start+1):stop] = y_pred

                        # ADAM Optimization
                        if adam:

                            beta1 = 0.9         # decay rate for the first moment
                            beta2 = 0.999       # decay rate for the second moment

                            # Update biased first and second moment estimate
                            m = beta1 * m + (1 - beta1) * gradients
                            v = beta2 * v + (1 - beta2) * (gradients ** 2)

                            # Compute bias-corrected first and second moment estimate
                            m_hat = m / (1 - beta1 ** epoch)
                            v_hat = v / (1 - beta2 ** epoch)

                            # Update control voltages given the gradients
                            control_voltages -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                        # Update control voltages given the gradients
                        else:
                            control_voltages -= learning_rate * gradients

                        # Print infos
                        if batch % print_nth_batch == 0:
                            print(f'Epoch   : {epoch}')
                            print(f'U_C     : {np.round(control_voltages,4)}')
                            print(f"Loss    : {loss}")

                # Save prediction
                if epoch % save_nth_epoch == 0:
                    np.savetxt(fname=f"{path}ypred_{epoch}.csv", X=predictions)
                    np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)

    def train_time_series_by_frequency(self, x : np.array, y : np.array, learning_rate : float, batch_size : int, N_epochs : int, epsilon=0.1, adam=False,
                          time_step=1e-10, stat_size=500, f_init=1.0, amplitude=0.1, transient_steps=0, print_nth_batch=1, save_nth_epoch=1, path=''):

        # Parameter
        N_voltages          = len(x)
        N_batches           = int(N_voltages/batch_size)
        N_controls          = self.N_electrodes - 2
        time_steps          = time_step*np.arange(N_voltages)
        params              = np.round(np.random.uniform(0.1, f_init, N_controls), 4) #np.repeat(f_init, N_controls)
        target_electrode    = self.N_electrodes - 1
        y                   = (y - np.mean(y))/np.std(y)

        # Multiprocessing Manager
        with multiprocessing.Manager() as manager:

            # Storage for simulation results
            return_dic = manager.dict()
            current_charge_vector = None
            
            # ADAM Optimization
            if adam:
                m = np.zeros_like(params)
                v = np.zeros_like(params)

            for epoch in range(1, N_epochs+1):
                
                # Set charge vector to None
                predictions = np.zeros(N_voltages)
                
                for batch in range(N_batches):

                    start   = batch*batch_size
                    stop    = (batch+1)*batch_size
                    
                    # Voltage array containing input at column 0 
                    voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
                    voltages[:,0]       = x[start:stop]
                    for i, f in enumerate(params):
                        voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
                    voltages_list       = []
                    voltages_list.append(voltages)

                    # Set up a list of voltages considering small deviations
                    for i in range(N_controls):

                        voltages_tmp        = voltages.copy()
                        voltages_tmp[:,i+1] = amplitude*np.cos((f+epsilon)*time_steps[start:stop]*1e8)
                        voltages_list.append(voltages_tmp)

                        voltages_tmp        = voltages.copy()
                        voltages_tmp[:,i+1] = amplitude*np.cos((f-epsilon)*time_steps[start:stop]*1e8)
                        voltages_list.append(voltages_tmp)

                    # Container for processes
                    procs = []

                    # For each set of voltages assign start a process
                    for thread in range(2*N_controls+1):

                        # Make a deep copy of the current instance for each process
                        instance_copy   = copy.deepcopy(self)

                        # Start process
                        process = multiprocessing.Process(target=self.sim_run_for_gradient_decent, args=(thread, instance_copy, return_dic, voltages_list[thread], time_steps,
                                                                                                    target_electrode, stat_size, current_charge_vector))
                        process.start()
                        procs.append(process)
                    
                    # Wait for all processes
                    for p in procs:
                        p.join()

                    # Current charge vector given the last iteration
                    current_charge_vector = return_dic[-1]
                    
                    # Gradient Container
                    gradients = np.zeros_like(params)

                    # Calculate gradients for each frequency 
                    for i in np.arange(1,2*N_controls+1,2):

                        y_pred_eps_pos          = return_dic[i]
                        y_pred_eps_neg          = return_dic[i+1]
                        y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
                        y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
                        perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
                        perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
                        gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                    # Current prediction and loss
                    y_pred  = return_dic[0]
                    y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
                    loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                    
                    predictions[(start+1):stop] = y_pred

                    # ADAM Optimization
                    if adam:

                        beta1 = 0.9         # decay rate for the first moment
                        beta2 = 0.999       # decay rate for the second moment

                        # Update biased first and second moment estimate
                        m = beta1 * m + (1 - beta1) * gradients
                        v = beta2 * v + (1 - beta2) * (gradients ** 2)

                        # Compute bias-corrected first and second moment estimate
                        m_hat = m / (1 - beta1 ** epoch)
                        v_hat = v / (1 - beta2 ** epoch)

                        # Update frequencies given the gradients
                        params -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                    # Update frequencies given the gradients
                    else:
                        params -= learning_rate * gradients

                    params = np.clip(params, 0.1, 10.0)

                    # Print infos
                    if batch % print_nth_batch == 0:
                        print(f'Epoch   : {epoch}')
                        print(f'U_C     : {np.round(params,4)}')
                        print(f"Loss    : {loss}")

                # Save prediction
                if epoch % save_nth_epoch == 0:
                    np.savetxt(fname=f"{path}ypred_{epoch}.csv", X=predictions)
                    np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)

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

class Optimizer(simulation):

    def __init__(self, optimizer_type : str, parameter_type: str, topology_parameter : dict, folder='', add_to_path="",
                 res_info=None, res_info2=None, np_info=None, np_info2=None, seed=None, **kwargs):

        super().__init__(topology_parameter, folder, add_to_path, res_info, res_info2, np_info, np_info2, seed, **kwargs)
        
        self.optimizer_type = optimizer_type
        self.parameter_type = parameter_type
        self.best_loss      = np.inf
        self.N_controls     = self.N_electrodes - 2

    def loss_function(self, y_pred : np.array, y_real : np.array, transient=0)->float:
        """Error given a norm.

        Parameters
        ----------
        y_pred : np.array
            Predicted values
        y_real : np.array
            Actual values
        transient : int, optional
            Neglect the first transient steps, by default 0

        Returns
        -------
        float
            RMSE
        """

        error = np.mean((y_pred[transient:]-y_real[transient:])**2)

        return error
    
    def simulate_current_loss(self, voltages : np.array, y_true : np.array, time_steps : np.array, stat_size=500, transient=0, init_charges=None):

        # Run Simulation
        target_electrode   = self.N_electrodes - 1
        self.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                              stat_size=stat_size, init_charges=init_charges, save=False)

        # Extract prediction and current loss
        output_values   = self.return_output_values()
        y_pred          = output_values[:,2]
        self.y_pred     = (y_pred - np.mean(y_pred)) / np.std(y_pred)
        self.loss       = self.loss_function(y_pred=y_pred, y_real=y_true, transient=transient)
    
    def simulate_current_loss_parallel(self, thread : int, class_instance, return_dic : dict, voltages : np.array, time_steps : np.array, stat_size=500, init_charges=None):
        """Simulation execution for gradient decent algorithm. Inits a class and runs simulation for variable voltages.

        Parameters
        ----------
        thread : int
            Thread if
        return_dic : dict
            Dictonary containing simulation results
        voltages : np.array
            Voltage values
        time_steps : np.array
            Time Steps
        target_electrode : int
            Electrode associated to target observable
        stat_size : int
            Number of individual runs for statistics
        network_topology : str
            Network topology, either 'cubic' or 'random'
        topology_parameter : dict
            Dictonary containing information about topology
        initial_charge_vector : _type_, optional
            If not None, initial_charge_vector is used as the first network state, by default None
        """

        
        target_electrode    = self.N_electrodes - 1
        class_instance.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
                                        stat_size=stat_size, init_charges=init_charges, save=False)
            
        # Get target observable 
        output_values       = class_instance.return_output_values()
        observable          = output_values[:,2]
        error_values        = output_values[:,3]
        return_dic[thread]  = observable

        if thread == 0:
            charge_values   = class_instance.q_eq
            return_dic[-1]  = charge_values

    def simulated_annealing(self, x : np.array, y : np.array, N_epochs : int, temp_init : float, cooling_rate=0.9, epsilon=0.01,
                            time_step=1e-10, stat_size=500, p_init=0.1, amplitude=0.02, freq=2.0, batch_size=0, print_nth_epoch=1, save_nth_epoch=1):

        # Target Values
        y           = (y - np.mean(y))/np.std(y)
        charge_init = None

        # Initial Values
        N_voltages  = len(x)

        if batch_size == 0:
            N_batches   = 1
            batch_size  = N_voltages
        else:
            N_batches   = N_voltages//batch_size

        if self.parameter_type == 'freq':
            self.best_params    = np.random.uniform(0.1, p_init, self.N_controls)
        elif self.parameter_type == 'phase':
            self.best_params    = np.random(0.0, 2*np.pi, self.N_controls)
        else:
            self.best_params    = np.random.uniform(-p_init, p_init, self.N_controls)

        time_steps  = time_step*np.arange(N_voltages)
        temp        = temp_init

        # For each epoch
        for epoch in range(1, N_epochs+1):

            for batch in range(N_batches):

                start   = batch*batch_size
                stop    = (batch+1)*batch_size
            
                # New parameter values
                params              = self.best_params + np.random.uniform(-epsilon, epsilon, self.N_controls)
                voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
                voltages[:,0]       = x[start:stop]

                if self.parameter_type == 'freq':
                    params = np.clip(params, a_min=0.1, a_max=10.0)
                    for i, f in enumerate(params):
                        voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
                elif self.parameter_type == 'phase':
                    for i, phi in enumerate(params):
                        voltages[:,i+1] = amplitude*np.cos(freq*time_steps[start:stop]*1e8-phi)
                else:
                    voltages[:,1:-2]    = np.tile(np.round(params,4), (batch_size,1))
            
                # Run Simulation and get new loss
                self.simulate_current_loss(voltages=voltages, y_true=y[start+1:stop], time_steps=time_steps[start:stop], stat_size=stat_size, transient=1000, init_charges=charge_init)
            
                delta_loss  = self.loss - self.best_loss
                charge_init = self.q_eq

                # Check for new best parameters
                if delta_loss < 0:
                    self.best_loss      = self.loss
                    self.best_params    = params
                else:
                    prob = min([1.0, np.exp(-delta_loss/temp)])
                    if np.random.rand() < prob:
                        self.best_loss      = self.loss
                        self.best_params    = params

                # Cool down temperature
                if cooling_rate == 0:
                    temp = temp/np.log(1+epoch)
                else:
                    temp = temp * cooling_rate

            # Print infos
            if epoch % print_nth_epoch == 0:
                print(f'Epoch   : {epoch}')
                print(f'U_C     : {np.round(self.best_params,4)}')
                print(f"Loss    : {self.best_loss}")
                print(f"Temp    : {temp}")

            # Save prediction
            if epoch % save_nth_epoch == 0:
                np.savetxt(fname=f"{self.folder}ypred_{epoch}.csv", X=self.y_pred)

    def gradient_decent(self, x : np.array, y : np.array, N_epochs : int, learning_rate : float, batch_size : int, epsilon=0.001, adam=False,
                        time_step=1e-10, stat_size=500, p_init=0.1, amplitude=0.02, transient_steps=0, print_nth_batch=1, save_nth_epoch=1):
        
        # Target Values
        y   = (y - np.mean(y))/np.std(y)

        # Initial Values
        N_voltages  = len(x)
        N_batches   = N_voltages//batch_size
        time_steps  = time_step*np.arange(N_voltages)

        if self.parameter_type == 'freq':
            params  = np.random.uniform(0.1, p_init, self.N_controls)
        else:
            params  = np.random.uniform(-p_init, p_init, self.N_controls)

        # Multiprocessing Manager
        with multiprocessing.Manager() as manager:

            # Storage for simulation results
            return_dic  = manager.dict()
            charge_init = None
            
            # ADAM Optimization
            if adam:
                m = np.zeros_like(params)
                v = np.zeros_like(params)

            for epoch in range(1, N_epochs+1):
                
                # Set charge vector to None
                predictions = np.zeros(N_voltages)
                
                for batch in range(N_batches):

                    start   = batch*batch_size
                    stop    = (batch+1)*batch_size
                    
                    # Voltage array containing input at column 0 
                    voltages        = np.zeros(shape=(batch_size, self.N_electrodes+1))
                    voltages[:,0]   = x[start:stop]
                    
                    if self.parameter_type == 'freq':
                        for i,f in enumerate(params):
                            voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
                        voltages_list       = []
                        voltages_list.append(voltages)

                        # Set up a list of voltages considering small deviations
                        for i,f in enumerate(params):

                            voltages_tmp        = voltages.copy()
                            voltages_tmp[:,i+1] = amplitude*np.cos((f+epsilon)*time_steps[start:stop]*1e8)
                            voltages_list.append(voltages_tmp)

                            voltages_tmp        = voltages.copy()
                            voltages_tmp[:,i+1] = amplitude*np.cos((f-epsilon)*time_steps[start:stop]*1e8)
                            voltages_list.append(voltages_tmp)
                    else:
                        voltages[:,1:-2]    = np.tile(np.round(params,4), (batch_size,1))
                        voltages_list       = []
                        voltages_list.append(voltages)

                        # Set up a list of voltages considering small deviations
                        for i in range(self.N_controls):

                            voltages_tmp        = voltages.copy()
                            voltages_tmp[:,i+1] += epsilon
                            voltages_list.append(voltages_tmp)

                            voltages_tmp        = voltages.copy()
                            voltages_tmp[:,i+1] -= epsilon
                            voltages_list.append(voltages_tmp)

                    # Container for processes
                    procs       = []

                    # For each set of voltages assign start a process
                    for thread in range(2*self.N_controls+1):
                        
                        self_copy = copy.deepcopy(self)

                        # Start process
                        process = multiprocessing.Process(target=self.simulate_current_loss_parallel, args=(thread, self_copy, return_dic, voltages,
                                                                                                            time_steps, stat_size, charge_init))
                        process.start()
                        procs.append(process)
                    
                    # Wait for all processes
                    for p in procs:
                        p.join()

                    # Current charge vector given the last iteration
                    charge_init = return_dic[-1]
                    
                    # Gradient Container
                    gradients = np.zeros_like(params)

                    # Calculate gradients for each control voltage 
                    for i in np.arange(1,2*self.N_controls+1,2):

                        y_pred_eps_pos          = return_dic[i]
                        y_pred_eps_neg          = return_dic[i+1]
                        y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
                        y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
                        perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
                        perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
                        gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                    
                    # if ((epoch != 1) and (batch != 0)):
                        
                    # Current prediction and loss
                    y_pred  = return_dic[0]
                    y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
                    loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                    
                    predictions[(start+1):stop] = y_pred

                    # ADAM Optimization
                    if adam:

                        beta1 = 0.9         # decay rate for the first moment
                        beta2 = 0.999       # decay rate for the second moment

                        # Update biased first and second moment estimate
                        m = beta1 * m + (1 - beta1) * gradients
                        v = beta2 * v + (1 - beta2) * (gradients ** 2)

                        # Compute bias-corrected first and second moment estimate
                        m_hat = m / (1 - beta1 ** epoch)
                        v_hat = v / (1 - beta2 ** epoch)

                        # Update control voltages given the gradients
                        params -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

                    # Update control voltages given the gradients
                    else:
                        params -= learning_rate * gradients

                    # Print infos
                    if batch % print_nth_batch == 0:
                        print(f'Epoch   : {epoch}')
                        print(f'U_C     : {np.round(params,4)}')
                        print(f"Loss    : {loss}")

                # Save prediction
                if epoch % save_nth_epoch == 0:
                    np.savetxt(fname=f"{self.folder}ypred_{epoch}.csv", X=predictions)
                    # np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)

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