import tunneling
import numpy as np
import pandas as pd
import os.path
from typing import List
from numba.experimental import jitclass
from numba import int64, float64, boolean, types

spec = [
    ('charge_vector', float64[::1]),
    ('potential_vector', float64[::1]),
    ('tunnel_rates', float64[::1]),
    ('co_tunnel_rates', float64[::1]),
    ('inv_capacitance_matrix', float64[:,::1]),
    ('const_capacitance_values', float64[::1]),
    ('const_capacitance_values_co1', float64[::1]),
    ('const_capacitance_values_co2', float64[::1]),
    ('C_np_target', float64),
    ('C_np_self', float64),
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
    ('rel_error', float64),
    ('jump_diff_mean', float64),
    ('jump_diff_mean2', float64),
    ('jump_diff_std', float64),
    ('total_jumps', int64),
    ('jump', int64),
    ('cotunneling', boolean),
    ('co_event_selected', boolean),
    ('zero_T', boolean),
    ('charge_mean', float64[::1]),
    ('potential_mean', float64[::1]),
    ('jump_storage', float64[::1]),
    ('jump_storage_co', float64[::1]),
    ('jump_dist_per_it', float64[:,::1]),
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
    jump_diff_mean : float
        Difference in Jumps towards/from target electrode
    jump_diff_mean2 : float
        Difference in Jumps towards/from target electrode
    jump_diff_std : float
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
                co_adv_index2, co_adv_index3, N_electrodes, N_particles, C_np_target, C_np_self):
        
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
        self.C_np_target                    = C_np_target
        self.C_np_self                      = C_np_self
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
        self.counter_output_jumps_pos    = 0      
        self.counter_output_jumps_neg    = 0      
        self.total_jumps                 = 0      
        self.time                        = 0.0    
        self.jump_diff_mean              = 0.0    
        self.jump_diff_mean2             = 0.0   
        self.jump_diff_std               = 0.0
        self.rel_error                   = 1.0 
        self.jump                        = 0

        # Storages
        self.charge_mean        = np.zeros(len(charge_vector))
        self.potential_mean     = np.zeros(len(potential_vector))
        self.jump_storage       = np.zeros(len(adv_index_rows))
        self.jump_storage_co    = np.zeros(len(co_adv_index1))

        # Co-tunneling bools
        self.cotunneling        = False
        self.co_event_selected  = False
        self.zero_T             = False
        
        if (np.sum(self.temperatures) == 0.0):
            self.zero_T = True        

        if (len(self.co_adv_index1) != 0):
            self.cotunneling = True
        
    def calc_potentials(self):
        """
        Compute potentials via matrix vector multiplication
        """

        self.potential_vector[self.N_electrodes:] = np.dot(self.inv_capacitance_matrix, self.charge_vector)

    def update_potentials(self, np1, np2):
        """
        Update potentials after occured jump

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
    
    def update_floating_electrode(self, target_electrode : int, idx_np_target)->None:

        self.potential_vector[target_electrode] = (self.C_np_target/self.C_np_self)*self.potential_vector[idx_np_target]

    def calc_tunnel_rates(self):
        """
        Compute tunnel rates
        """

        free_energy         = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        self.tunnel_rates   = (free_energy/self.resistances)/(np.exp(free_energy/self.temperatures) - 1.0)

    def calc_tunnel_rates_zero_T(self):
        """
        Compute tunnel rates in zero T approximation
        """
                
        free_energy = self.ele_charge*(self.potential_vector[self.adv_index_cols] - self.potential_vector[self.adv_index_rows]) + self.const_capacitance_values
        
        self.tunnel_rates                   = np.zeros(self.N_rates)
        self.tunnel_rates[free_energy<0]    = -free_energy[free_energy<0]/self.resistances[free_energy<0]

    def calc_cotunnel_rates(self):
        """
        Compute cotunnel rates
        """

        free_energy1    = self.ele_charge*(self.potential_vector[self.co_adv_index1] - self.potential_vector[self.co_adv_index2]) + self.const_capacitance_values_co1
        free_energy2    = self.ele_charge*(self.potential_vector[self.co_adv_index2] - self.potential_vector[self.co_adv_index3]) + self.const_capacitance_values_co2

        factor          = self.planck_const/(3*np.pi*(self.ele_charge**4)*self.resistances_co1*self.resistances_co2)
        val1            = free_energy2/(4*free_energy1**2 - 4*free_energy1*free_energy2 + free_energy2**2)
        val2            = ((2*np.pi*self.temperatures_co)**2 + free_energy2**2)/(np.exp(free_energy2/self.temperatures_co) - 1.0)
        
        self.co_tunnel_rates = factor*val1*val2
    
    def calc_cotunnel_rates_zero_T(self):
        """
        Compute cotunnel rates in zero T approximation
        """

        free_energy1    = self.ele_charge*(self.potential_vector[self.co_adv_index1] - self.potential_vector[self.co_adv_index2]) + self.const_capacitance_values_co1
        free_energy2    = self.ele_charge*(self.potential_vector[self.co_adv_index2] - self.potential_vector[self.co_adv_index3]) + self.const_capacitance_values_co2

        factor                  = self.planck_const/(3*np.pi*(self.ele_charge**4)*self.resistances_co1*self.resistances_co2)
        val1                    = free_energy2/(4*free_energy1**2 - 4*free_energy1*free_energy2 + free_energy2**2)
        val2                    = np.zeros(self.N_corates)
        val2[free_energy2<0]    = -free_energy2[free_energy2<0]**2

        self.co_tunnel_rates = factor*val1*val2
    
    def calc_rel_error(self, N_calculations):
        """
        Calculate relative error and standard deviation by welford one pass 
        """

        # if (self.jump_diff_mean != 0.0):
        if N_calculations >= 2:

            self.jump_diff_std   = np.sqrt(np.abs(self.jump_diff_mean2) / (N_calculations - 1))/np.sqrt(N_calculations)
            # self.jump_diff_std   = np.sqrt(np.abs(self.jump_diff_mean2) / self.total_jumps)/np.sqrt(N_calculations)
            self.rel_error       = self.jump_diff_std/np.abs(self.jump_diff_mean)
    
    def select_event(self, random_number1 : float, random_number2 : float):
        """
        Select next charge hopping event and update time by kinetic monte carlo apporach.
        Updates given charge vector based on executed event
        
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
            
            self.time   = 1.0
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

    def neglect_last_event(self, np1,np2):

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
    
    def select_co_event(self, random_number1 : float, random_number2 : float):
        """
        Select next charge hopping event and update time by kinetic monte carlo apporach considering cotunneling
        Updates given charge vector based on executed event
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

        self.calc_potentials()

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
            
        return n_jumps
    
    def kmc_simulation_potential(self, target_electrode : int, error_th=0.05, max_jumps=10000000, jumps_per_stat=1000):

        count                   = 0
        total_time              = 0
        self.total_jumps        = 0
        self.rel_error          = 1.0
        self.jump_diff_mean     = 0.0
        self.jump_diff_mean2    = 0.0
        self.jump_diff_std      = 0.0
        idx_np_target           = self.adv_index_cols[target_electrode]

        # Initial potential landscape
        self.calc_potentials()

        # Until the desired relative error is reached or max_jumps is exceeded
        while((self.rel_error > error_th) and (self.total_jumps < max_jumps)):

            self.time           = 0.0
            target_potential    = 0.0

            for i in range(jumps_per_stat):

                # KMC Iteration
                random_number1  = np.random.rand()
                random_number2  = np.random.rand()

                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()

                self.select_event(random_number1, random_number2)
                self.jump_storage[self.jump] += 1

                # Track charge and potential landscape
                self.charge_mean    += self.charge_vector
                self.potential_mean += self.potential_vector
                
                # Update potential of floating target electrode
                self.update_floating_electrode(target_electrode, idx_np_target)
                # target_potentials[i] = self.potential_vector[target_electrode]
                target_potential += self.potential_vector[target_electrode]

            # Update total time and jumps
            total_time          += self.time
            self.total_jumps    += jumps_per_stat

            if self.time != 0:
                # Calc new average electrode potential
                jump_diff_new = target_potential/jumps_per_stat
                self.jump_diff_mean, self.jump_diff_mean2, count = self.return_next_means(jump_diff_new, self.jump_diff_mean, self.jump_diff_mean2, count)

            # Update realative error
            if ((self.jump_diff_mean != 0) and (self.jump != -1)):

                self.calc_rel_error(count)

            elif ((self.jump_diff_mean == 0) and (self.jump != -1)):
            
                count -= 1

            else:
                
                self.jump_diff_mean = 0.0
                break
    
    def return_next_means(self, new_value, mean_value, mean_value2, count):

        count           +=  1
        delta           =   new_value - mean_value
        mean_value      +=  delta/count          
        delta2          =   new_value - mean_value
        mean_value2     +=  delta * delta2

        return mean_value, mean_value2, count
    
    def kmc_simulation_fixed(self, target_electrode : int, error_th = 0.05, max_jumps=10000000, jumps_per_stat=1000):
        """
        Runs KMC until current for target electrode has a relative error below error_th or max_jumps is reached
        Tracks Mean Current, Current standard deviation, average microstate (charge vector), average landscape (potential vector), contribution of each junction

        Parameters
        ----------
        target_electrode : int
            electrode index of which electric current is estimated
        error_th : float
            relative error in current to be achieved
        max_jumps : int
            maximum number of jumps before function ends
        jumps_per_stat : int
            number of jumps for each current sample
        """
        
        count                           = 0
        total_time                      = 0
        self.total_jumps                = 0
        self.rel_error                  = 1.0
        self.jump_diff_mean             = 0.0
        self.jump_diff_mean2            = 0.0
        self.jump_diff_std              = 0.0
        np1, np2                        = self.adv_index_cols[self.jump], self.adv_index_rows[self.jump]  

        self.calc_potentials()

        while((self.rel_error > error_th) and (self.total_jumps < max_jumps)):

            # Reset tracked observables
            self.counter_output_jumps_pos   = 0
            self.counter_output_jumps_neg   = 0
            self.time                       = 0.0
            jump_storage_vals               = np.zeros(len(self.adv_index_rows))

            for i in range(jumps_per_stat):
            
                # KMC Part
                random_number1  = np.random.rand()
                random_number2  = np.random.rand()

                if self.cotunneling == False:

                    if not(self.zero_T):
                        self.calc_tunnel_rates()
                    else:
                        self.calc_tunnel_rates_zero_T()
                    self.select_event(random_number1, random_number2)

                    np1 = self.adv_index_rows[self.jump]
                    np2 = self.adv_index_cols[self.jump]
                    # self.jump_storage[self.jump]    += 1
                    jump_storage_vals[self.jump]    += 1

                else:
                    
                    if not(self.zero_T):
                        self.calc_tunnel_rates()
                        self.calc_cotunnel_rates()
                    else:
                        self.calc_tunnel_rates_zero_T()
                        self.calc_cotunnel_rates_zero_T()

                    self.select_co_event(random_number1, random_number2)

                    if self.co_event_selected:
                        np1 = self.co_adv_index1[self.jump]
                        np2 = self.co_adv_index3[self.jump]
                        self.jump_storage_co[self.jump] += 1
                    else:
                        np1 = self.adv_index_rows[self.jump]
                        np2 = self.adv_index_cols[self.jump]

                    self.jump_storage[self.jump]    += 1

                self.charge_mean    += self.charge_vector
                self.potential_mean += self.potential_vector
                
                # If jump from target electrode
                if (np1 == target_electrode):
                    self.counter_output_jumps_neg += 1
                    
                # If jump towards target electrode
                if (np2 == target_electrode):
                    self.counter_output_jumps_pos += 1

            self.total_jumps    += i+1
            total_time          += self.time

            if self.time != 0:

                # Calc new average currents
                jump_diff_new = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
                self.jump_diff_mean, self.jump_diff_mean2, count = self.return_next_means(jump_diff_new, self.jump_diff_mean, self.jump_diff_mean2, count)
                
                self.jump_storage += jump_storage_vals/self.time

            # Update realative error
            if ((self.jump_diff_mean != 0) and (self.jump != -1)):

                self.calc_rel_error(count)

            elif ((self.jump_diff_mean == 0) and (self.jump != -1)):
            
                count -= 1

            else:
                
                self.jump_diff_mean = 0.0
                break
              
        self.jump_storage = self.jump_storage/count
    
    def kmc_time_simulation(self, target_electrode : int, time_target : float, store_per_it_min=0, store_per_it_max=0):
        """
        Runs KMC until KMC time exceeds a target value

        Parameters
        ----------
        target_electrode : int
            electrode index of which electric current is estimated
        time_target : float
            time value to be reached
        """

        self.counter_output_jumps_neg   = 0
        self.counter_output_jumps_pos   = 0
        self.jump_storage               = np.zeros(len(self.adv_index_rows))
        self.charge_mean                = np.zeros(len(self.charge_vector))
        self.potential_mean             = np.zeros(len(self.potential_vector))
        self.jump_dist_per_it           = np.expand_dims(np.zeros(len(self.adv_index_rows)),0)
        self.landscape_per_it           = np.expand_dims(np.zeros(len(self.potential_vector)),0)
        self.time_vals                  = np.zeros(1)
        self.total_jumps                = 0
        inner_time                      = self.time
        last_time                       = 0.0

        while (self.time < time_target):

            # KMC Part
            random_number1  = np.random.rand()
            random_number2  = np.random.rand()
            last_time       = self.time

            if not(self.zero_T):
                self.calc_tunnel_rates()
            else:
                self.calc_tunnel_rates_zero_T()
            self.select_event(random_number1, random_number2)

            if (self.jump == -1):
                break

            np1 = self.adv_index_rows[self.jump]
            np2 = self.adv_index_cols[self.jump]

            if (self.time >= time_target):
                self.neglect_last_event(np1,np2)
                break

            self.jump_storage[self.jump]    += 1
            
            self.charge_mean    += self.charge_vector
            self.potential_mean += self.potential_vector
            
            if ((self.time >= store_per_it_min) & (self.time < store_per_it_max)):
                self.jump_dist_per_it   = np.vstack((self.jump_dist_per_it, np.expand_dims(self.jump_storage,0)))
                self.landscape_per_it   = np.vstack((self.landscape_per_it, np.expand_dims(self.potential_vector,0)))
                self.time_vals          = np.hstack((self.time_vals, np.array([self.time])))

            # If jump from target electrode
            if (np1 == target_electrode):
                self.counter_output_jumps_neg += 1
            # If jump towards target electrode
            if (np2 == target_electrode):
                self.counter_output_jumps_pos += 1
            
            # Statistics
            self.total_jumps +=  1
        
        if (self.jump == -1):

            self.jump_diff_mean = 0.0
            self.jump_storage   = self.jump_storage/(time_target-inner_time)

        if (last_time-inner_time) != 0:
            
            self.jump_diff_mean = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/(time_target-inner_time)
            self.jump_storage   = self.jump_storage/(time_target-inner_time)

        else:
            self.jump_diff_mean = 0
        
    def update_bimodal_resistance(self, I_tilde, slope : float, shift : float, R_max=30, R_min=20):

        R1 = R_max*self.ele_charge*self.ele_charge*1e-12
        R2 = R_min*self.ele_charge*self.ele_charge*1e-12

        self.resistances = (R1 - R2)*(-np.tanh(slope*(I_tilde - shift)) + 1)/2 + R2

    def kmc_simulation_var_resistance(self, target_electrode : int, error_th=0.05, max_jumps=10000000, jumps_per_stat=1000,
                                      slope=1, shift=1, tau_0=1, R_max=30, R_min=20, verbose=False):
        
        count                           = 0
        total_time                      = 0
        self.total_jumps                = 0
        self.rel_error                  = 1.0
        self.jump_diff_mean             = 0.0
        self.jump_diff_mean2            = 0.0
        self.jump_diff_std              = 0.0
        np1, np2                        = self.adv_index_cols[self.jump], self.adv_index_rows[self.jump]
        I_tilde                         = np.zeros(len(self.adv_index_rows))

        # For additional information and without time scale bins
        if (verbose and (jumps_per_stat == max_jumps)):

            self.landscape_per_it   = np.zeros((max_jumps, len(self.potential_vector)))
            self.resistances_per_it = np.zeros((max_jumps, len(self.adv_index_rows)))
            self.jump_dist_per_it   = np.zeros((max_jumps, len(self.adv_index_rows)))
            self.time_vals          = np.zeros(max_jumps)
        
        self.calc_potentials()        

        while((self.rel_error > error_th) and (self.total_jumps < max_jumps)):

            # Reset tracked observables
            self.counter_output_jumps_pos   = 0
            self.counter_output_jumps_neg   = 0
            self.time                       = 0.0
            jump_storage_vals               = np.zeros(len(self.adv_index_rows))
            
            for i in range(jumps_per_stat):
                
                # Time before event
                t1  = self.time
                
                # Select an event based on rates
                random_number1  = np.random.rand()
                random_number2  = np.random.rand()

                if not(self.zero_T):
                    self.calc_tunnel_rates()
                else:
                    self.calc_tunnel_rates_zero_T()
                
                self.select_event(random_number1, random_number2) 

                # Selected event, i.e. nanoparticles
                np1 = self.adv_index_rows[self.jump]
                np2 = self.adv_index_cols[self.jump]
                jump_storage_vals[self.jump]    += 1

                # Time after event
                t2                  = self.time
                dt                  = t2 - t1
                
                # Current contribution to resistance
                I_tilde             = I_tilde*np.exp(-dt/tau_0)
                I_tilde[self.jump]  += 1

                # New resistances
                self.update_bimodal_resistance(I_tilde, slope, shift, R_max, R_min)
                
                # If additional informations are required, track observables
                if (verbose and (jumps_per_stat == max_jumps)):

                    self.landscape_per_it[i,:]      = self.potential_vector
                    self.resistances_per_it[i,:]    = self.resistances
                    self.jump_dist_per_it[i,:]      = jump_storage_vals
                    self.time_vals[i,:]             = t2
                
                # Update average charge and potential distribution
                self.charge_mean    += self.charge_vector
                self.potential_mean += self.potential_vector
                
                # If jump from target electrode
                if (np1 == target_electrode):
                    self.counter_output_jumps_neg += 1
                    
                # If jump towards target electrode
                if (np2 == target_electrode):
                    self.counter_output_jumps_pos += 1

            self.total_jumps    += i+1
            total_time          += self.time

            if self.time != 0:

                # Calc new average currents
                jump_diff_new = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
                self.jump_diff_mean, self.jump_diff_mean2, count = self.return_next_means(jump_diff_new, self.jump_diff_mean, self.jump_diff_mean2, count)

                self.jump_storage += jump_storage_vals/self.time

            # Update realative error
            if ((self.jump_diff_mean != 0) and (self.jump != -1)):

                self.calc_rel_error(count)

            elif ((self.jump_diff_mean == 0) and (self.jump != -1)):

                count -= 1

            else:
                
                self.jump_diff_mean = 0.0
                break

        self.jump_storage = self.jump_storage/count
    
    
    def return_target_values(self):
        """
        Returns
        -------
        jump_diff_mean : float
            Difference in target jumps towards/from target electrode
        jump_diff_std : float
            Standard Deviation for difference in target jumps
        self.charge_mean/self.total_jumps : array
            Average charge landscape
        self.jump_storage
            Contribution of all tunnel junctions
        self.jump_storage_co
            Contribution of all cotunnel junctions
        self.total_jumps
            Number of total jumps
        """

        if self.total_jumps != 0:
        
            return self.jump_diff_mean, self.jump_diff_std, self.charge_mean/self.total_jumps,  self.potential_mean/self.total_jumps, self.jump_storage, self.jump_storage_co, self.landscape_per_it, self.jump_dist_per_it, self.time_vals, self.total_jumps

        else:

            return self.jump_diff_mean, self.jump_diff_std, self.charge_mean,  self.potential_mean, self.jump_storage, self.jump_storage_co, self.landscape_per_it, self.jump_dist_per_it, self.time_vals, self.total_jumps

###################################################################################################
# FUNCTIONS
###################################################################################################

def save_target_currents(output_values : List[np.array], voltages : np.array, path : str)->None:
    
    data    = np.hstack((voltages, output_values))
    columns = [f'E{i}' for i in range(voltages.shape[1]-1)]
    columns = np.array(columns + ['G', 'Eq_Jumps', 'Jumps', 'Current', 'Error'])

    df          = pd.DataFrame(data)
    df.columns  = columns

    df['Current']   = df['Current']*10**(-6)
    df['Error']   = df['Error']*10**(-6)

    if (os.path.isfile(path)):

        df.to_csv(path, mode='a', header=False, index=False)

    else:
        df.to_csv(path, header=True, index=False)

def save_mean_microstate(microstates : List[np.array], path : str)->None:

    ele_charge     = 0.160217662
    microstates_df = pd.DataFrame(microstates)/ele_charge

    if (os.path.isfile(path)):
        microstates_df.to_csv(path, mode='a', header=False, index=False)
    else:
        microstates_df.to_csv(path, header=True, index=False)

def save_jump_storage(average_jumps : List[np.array], adv_index_rows : np.array, adv_index_cols : np.array, path : str)->None:

    avg_j_cols                  = [(adv_index_rows[i],adv_index_cols[i]) for i in range(len(adv_index_rows))]
    average_jumps_df            = pd.DataFrame(average_jumps)
    average_jumps_df.columns    = avg_j_cols
    average_jumps_df            = average_jumps_df*10**(-6)

    if (os.path.isfile(path)):
        average_jumps_df.to_csv(path, mode='a', header=False, index=False)
    else:
        average_jumps_df.to_csv(path, header=True, index=False)

def save_cojump_storage(average_cojumps : List[np.array], co_adv_index1 : np.array, co_adv_index3 : np.array, path : str)->None:

    avg_j_cols                  = [(co_adv_index1[i],co_adv_index3[i]) for i in range(len(co_adv_index1))]
    average_jumps_df            = pd.DataFrame(average_cojumps)
    average_jumps_df.columns    = avg_j_cols
    average_jumps_df            = average_jumps_df*10**(-6)

    if (os.path.isfile(path)):
        average_jumps_df.to_csv(path, mode='a', header=False, index=False)
    else:
        average_jumps_df.to_csv(path, header=True, index=False)

#####################################################################################################################################################################################
#####################################################################################################################################################################################

class simulation(tunneling.tunnel_class):

    def __init__(self, network_topology : str, topology_parameter : dict, folder='', res_info=None, res_info2=None, np_info=None, np_info2=None,
                    add_to_path="", del_n_junctions=0, gate_nps=None, tunnel_order=1, seed=None):
        """Defines network topology, electrostatic properties and tunneling junctions for a given type of topology. 

        Parameters
        ----------
        network_topology : str
            Type of network topology. Either use "cubic" to set up a lattice topology or "random" to set up a disordered network topology 
        topology_parameter : dict
            Dictonary including information about number of nanoparticles and electrode positions.
        folder : str, optional
            Folder where simulation results are saved, by default ''
        res_info : dict, optional
            Dictonary including information about resistance values for the first type of nanoparticles, by default None
        res_info2 : dict, optional
            Dictonary including information about resistance values for the second type of nanoparticles, by default None
        np_info : dict, optional
            Dictonary including information about the first type of nanoparticles, by default None
        np_info2 : dict, optional
            Dictonary including information about the second type of nanoparticles, by default None
        add_to_path : str, optional
            String which is extended to the file path, by default ""
        del_n_junctions : int, optional
            Randomly delete n junctions in a cubic network topology, by default 0
        gate_nps : _type_, optional
            Define which nanoparticles are affected by a gate electrode, by default None
        tunnel_order : int, optional
            Value either set 1 or 2 to specify whether the simulation considers only first order or also second order tunneling event, by default 1
        seed : _type_, optional
            Seed in use, when network properties are randomly sampled, by default None

        Raises
        ------
        ValueError
            _description_
        """

        super().__init__(tunnel_order, seed)

        # Type of Network Topology:
        self.network_topology = network_topology

        # Parameter of first nanoparticle type
        if np_info == None:
            np_info = {
                "eps_r"         : 2.6,  # Permittivity of molecular junction 
                "eps_s"         : 3.9,  # Permittivity of oxide layer
                "mean_radius"   : 10.0, # average nanoparticle radius
                "std_radius"    : 0.0,  # standard deviation of nanoparticle radius
                "np_distance"   : 1.0   # spacing between nanoparticle shells
            }

        # First type of nanoparticle resistances
        if res_info == None:
            res_info = {
                "mean_R"    : 25.0, # Average resistance
                "std_R"     : 0.0,  # Standard deviation of resistances
                "dynamic"   : False # Dynamic or constant resistances
            }
        
        self.res_info   = res_info
        self.res_info2  = res_info2

        # For a cubic topology
        if network_topology == "cubic":

            # Path variable
            path_var = f'Nx={topology_parameter["Nx"]}_Ny={topology_parameter["Ny"]}_Nz={topology_parameter["Nz"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # Cubic Network Topology
            self.cubic_network(N_x=topology_parameter["Nx"], N_y=topology_parameter["Ny"], N_z=topology_parameter["Nz"])
            self.set_electrodes_based_on_pos(topology_parameter["e_pos"], topology_parameter["Nx"], topology_parameter["Ny"])
            self.attach_np_to_gate(gate_nps=gate_nps)
            
            # Delete Junctions if porvided
            if del_n_junctions != 0:
                self.delete_n_junctions(del_n_junctions)
                
            # Electrostatic Properties
            self.init_nanoparticle_radius(np_info['mean_radius'], np_info['std_radius'])

            # Second Type of Nanopartciles
            if np_info2 != None:
                self.update_nanoparticle_radius(np_info2['np_index'], np_info2['mean_radius'], np_info2['std_radius'])

            # Capacitance Matrix
            self.calc_capacitance_matrix(np_info['eps_r'], np_info['eps_s'], np_info['np_distance'])

        # For a disordered topology
        elif network_topology == "random":
            
            # Path variable
            path_var = f'Np={topology_parameter["Np"]}_Nj={topology_parameter["Nj"]}_Ne={len(topology_parameter["e_pos"])}'+add_to_path+'.csv'
            
            # Random Network Topology
            self.random_network(N_particles=topology_parameter["Np"], N_junctions=topology_parameter["Nj"])
            self.add_electrodes_to_random_net(electrode_positions=topology_parameter["e_pos"])
            self.graph_to_net_topology()
            self.attach_np_to_gate(gate_nps=gate_nps)
                            
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
        self.path1  = folder + path_var
        self.path2  = folder + 'mean_state_' + path_var
        self.path3  = folder + 'net_currents_' + path_var

        # Simulation Obseravles
        self.output_values   = []
        self.microstates     = []
        self.landscape       = []
        self.pot_values      = []
        self.jumps_per_it    = []
        self.time_values     = []
        self.pot_per_it      = []
        self.average_jumps   = []
        self.average_cojumps = []

    def run_const_voltages(self, voltages : np.array, target_electrode : int, T_val=0.0, sim_dic=None, save_th=10, output_potential=False):
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
        """

        # Simulation Parameter
        if sim_dic != None:
            error_th        = sim_dic['error_th']
            max_jumps       = sim_dic['max_jumps']
            eq_steps        = sim_dic['eq_steps']
            jumps_per_stat  = sim_dic['jumps_per_stat']
        else:
            error_th        = 0.05
            max_jumps       = 10000000
            eq_steps        = 100000
            jumps_per_stat  = 1000

        j = 0
        
        for i, voltage_values in enumerate(voltages):
            
            # Based on current voltages get charges and potentials
            self.init_charge_vector(voltage_values=voltage_values)
            self.init_potential_vector(voltage_values=voltage_values)
            self.init_const_capacitance_values()
            self.np_target_electrode_electrostatic_properties(target_electrode)

            # Return Model Arguments
            inv_capacitance_matrix                                                                  = self.return_inv_capacitance_matrix()
            charge_vector                                                                           = self.return_charge_vector()
            potential_vector                                                                        = self.return_potential_vector()
            const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2    = self.return_const_capacitance_values()
            N_electrodes, N_particles                                                               = self.return_particle_electrode_count()
            adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2, co_adv_index3             = self.return_advanced_indices()
            temperatures, temperatures_co                                                           = self.return_const_temperatures(T=T_val)
            resistances, resistances_co1, resistances_co2                                           = self.return_random_resistances(R=self.res_info['mean_R'], Rstd=self.res_info['std_R'])
            idx_np_target, C_np_self, C_np_target                                                   = self.return_output_electrostatics()

            
            if self.res_info2 != None:
                resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

            # Pass all model arguments into Numba optimized Class
            model = model_class(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values, const_capacitance_values_co1, const_capacitance_values_co2,
                                    temperatures, temperatures_co, resistances, resistances_co1, resistances_co2, adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2,
                                    co_adv_index3, N_electrodes, N_particles, C_np_target, C_np_self)

            # Eqilibrate Potential Landscape
            eq_jumps = model.run_equilibration_steps(eq_steps)

            # Production Run until Current at target electrode is less than error_th or max_jumps was passed
            if output_potential:
                model.kmc_simulation_potential(target_electrode, error_th, max_jumps, jumps_per_stat)
            else:
                model.kmc_simulation_fixed(target_electrode, error_th, max_jumps, jumps_per_stat)
            
            jump_diff_mean, jump_diff_std, mean_state, mean_potentials, executed_jumps, executed_cojumps, landscape_per_it, jump_dist_per_it, time_vals, total_jumps = model.return_target_values()
            
            # Append Results to Outputs
            self.output_values.append(np.array([eq_jumps, total_jumps, jump_diff_mean, jump_diff_std]))
            self.microstates.append(mean_state)
            self.landscape.append(mean_potentials)
            self.average_jumps.append(executed_jumps)
            self.average_cojumps.append(executed_cojumps)

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

    def run_var_voltages(self, voltages : np.array, time_steps : np.array, target_electrode, eq_steps=100000, T_val=0.0,
                         save_th=10, store_per_it_min=0, store_per_it_max=0, init=True):

        if init:
            
            # First time step
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
            
            if self.res_info2 != None:
                resistances = self.update_nanoparticle_resistances(resistances, self.res_info2["np_index"], self.res_info2["R"])

            # Simulation Class
            self.model = model_class(charge_vector, potential_vector, inv_capacitance_matrix, const_capacitance_values, const_capacitance_values_co1,const_capacitance_values_co2,
                                    temperatures, temperatures_co, resistances, resistances_co1, resistances_co2, adv_index_rows, adv_index_cols, co_adv_index1, co_adv_index2,
                                    co_adv_index3, N_electrodes, N_particles)

            # Eqilibrate Potential Landscape
            eq_jumps = self.model.run_equilibration_steps(eq_steps)

            # Initial time and Jumps towards and from target electrode
            self.model.time                     = 0.0
            self.model.counter_output_jumps_neg = 0
            self.model.counter_output_jumps_pos = 0

            # Subtract charges induces by initial electrode voltages
            offset                      = self.get_charge_vector_offset(voltage_values=voltages[0])
            self.model.charge_vector    = self.model.charge_vector - offset
        
        else:
            eq_jumps = 0
            
        j = 0
        self.output_values   = []
        self.microstates     = []
        self.landscape       = []
        self.pot_values      = []
        self.jumps_per_it    = []
        self.time_values     = []
        self.pot_per_it      = []
        self.average_jumps   = []
        self.average_cojumps = []

        # self.charge_values_pos   = []
        # self.charge_values_neg   = []

        for i, voltage_values in enumerate(voltages[:-1]):
            
            offset                      = self.get_charge_vector_offset(voltage_values=voltage_values)
            self.model.charge_vector    = self.model.charge_vector + offset
            
            self.model.time = time_steps[i]
            time_target     = time_steps[i+1]

            # Update Electrode Potentials
            self.model.potential_vector[:(len(voltage_values)-1)]  = voltage_values[:(len(voltage_values)-1)]
            self.model.kmc_time_simulation(target_electrode, time_target, store_per_it_min, store_per_it_max)
            jump_diff_mean, jump_diff_std, mean_state, mean_potentials, executed_jumps, executed_cojumps, landscape_per_it, jump_dist_per_it, time_vals, total_jumps = self.model.return_target_values()
            
            # Append Results to Outputs
            self.output_values.append(np.array([eq_jumps, total_jumps, jump_diff_mean, jump_diff_std]))
            self.microstates.append(mean_state)
            self.landscape.append(mean_potentials)
            self.average_jumps.append(executed_jumps)
            self.average_cojumps.append(executed_cojumps)
            self.jumps_per_it.append(jump_dist_per_it)
            self.pot_per_it.append(landscape_per_it)
            self.time_values.append(time_vals)

            # self.charge_values_pos.append(self.model.counter_output_jumps_pos)
            # self.charge_values_neg.append(self.model.counter_output_jumps_neg)
            # self.pot_values.append()

            # Store Data
            if ((i+1) % save_th == 0):
                
                save_target_currents(np.array(self.output_values), voltages[j:(i+1),:], self.path1)
                save_mean_microstate(self.microstates, self.path2)
                save_jump_storage(self.average_jumps, adv_index_rows, adv_index_cols, self.path3)
                self.output_values      = []
                self.microstates        = []
                self.landscape          = []
                self.average_jumps      = []
                self.average_cojumps    = []
                j                       = i+1
            
            offset                      = self.get_charge_vector_offset(voltage_values=voltage_values)
            self.model.charge_vector    = self.model.charge_vector - offset

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
        return_var[:,2] = return_var[:,2]*10**(-6)
        return_var[:,3] = return_var[:,3]*10**(-6)

        return return_var

    def return_microstates(self):

        return np.array(self.microstates)

    def return_potential_landscape(self):

        return np.array(self.landscape)

    def return_network_currents(self):

        avg_j_cols = [(self.adv_index_rows[i],self.adv_index_cols[i]) for i in range(len(self.adv_index_rows))]

        return avg_j_cols, np.array(self.average_jumps)*10**(-6)   
    
    def return_jumps_per_it(self):

        return self.jumps_per_it
    
    def return_pot_per_it(self):

        return self.pot_per_it
    
    def return_time_vals(self):

        return self.time_values
    
###########################################################################################################################
###########################################################################################################################

if __name__ == '__main__':

    # Voltage Sweep
    N_voltages          = 4#80028
    N_processes         = 4
    v_rand              = np.repeat(np.random.uniform(low=-0.05, high=0.05, size=((int(N_voltages/4),5))), 4, axis=0)
    v_gates             = np.repeat(np.random.uniform(low=-0.1, high=0.1, size=int(N_voltages/4)),4)
    i1                  = np.tile([0.0,0.0,0.01,0.01], int(N_voltages/4))
    i2                  = np.tile([0.0,0.01,0.0,0.01], int(N_voltages/4))
    voltages            = pd.DataFrame(np.zeros((N_voltages,9)))
    voltages.iloc[:,0]  = v_rand[:,0]
    voltages.iloc[:,2]  = v_rand[:,1]
    voltages.iloc[:,4]  = v_rand[:,2]
    voltages.iloc[:,5]  = v_rand[:,3]
    voltages.iloc[:,6]  = v_rand[:,4]
    voltages.iloc[:,1]  = i1
    voltages.iloc[:,3]  = i2
    voltages.iloc[:,-1] = v_gates
    print(voltages)

    # folder                      = ""
    # topology_parameter          = {}
    # topology_parameter["Nx"]    = 3
    # topology_parameter["Ny"]    = 3
    # topology_parameter["Nz"]    = 1
    # topology_parameter["e_pos"] = [[0,0,0],[2,0,0],[0,2,0],[2,2,0]]
    # target_electrode            = len(topology_parameter['e_pos'])-1
        
    # sim_dic                 = {}
    # sim_dic['error_th']     = 0.05
    # sim_dic['max_jumps']    = 5000000

    # sim_class = simulation("", voltages=voltages.values, topology_parameter=topology_parameter)
    # sim_class.run_const_voltages(3, sim_dic=sim_dic, save_th=1)

    folder = ""
    topology_parameter          = {}
    topology_parameter["Np"]    = 20
    topology_parameter["Nj"]    = 4
    topology_parameter["e_pos"] = [[-1.5,-1.5],[0,-1.5],[1.5,-1.5],[-1.5,0],[-1.5,1.5],[1.5,0],[0,1.5],[1.5,1.5]]
    target_electrode            = len(topology_parameter['e_pos'])-1

    sim_class = simulation(voltages=voltages.values)
    sim_class.init_random(folder=folder, topology_parameter=topology_parameter)
    sim_class.run_const_voltages(target_electrode=target_electrode, save_th=1)