# def kmc_simulation(self, target_electrode : int, error_th = 0.05, max_jumps=10000000):
#         """
#         Runs KMC until current for target electrode has a relative error below error_th or max_jumps is reached
#         Tracks Mean Current, Current standard deviation, average microstate (charge vector), contribution of each junction

#         Parameters
#         ----------
#         target_electrode : int
#             electrode index of which electric current is estimated
#         error_th : float
#             relative error in current to be achieved
#         max_jumps : int
#             maximum number of jumps before function ends
#         """
        
#         N_calculations                  = 0
#         self.total_jumps                = 0
#         self.time                       = 0.0
#         self.rel_error                  = 1.0
#         self.counter_output_jumps_pos   = 0
#         self.counter_output_jumps_neg   = 0
#         self.jump_diff_mean             = 0.0
#         self.jump_diff_mean2            = 0.0
#         self.jump_diff_std              = 0.0
#         np1, np2                        = self.adv_index_cols[self.jump], self.adv_index_rows[self.jump]     
#         self.calc_potentials()

#         while(self.rel_error > error_th):

#             if ((self.total_jumps == max_jumps) or (self.jump == -1)):

#                 break
                        
#             # KMC Part
#             random_number1  = np.random.rand()
#             random_number2  = np.random.rand()

#             if self.cotunneling == False:
#                 if not(self.zero_T):
#                     self.calc_tunnel_rates()
#                 else:
#                     self.calc_tunnel_rates_zero_T()
#                 self.select_event(random_number1, random_number2)

#                 np1 = self.adv_index_rows[self.jump]
#                 np2 = self.adv_index_cols[self.jump]
#                 self.jump_storage[self.jump]    += 1

#             else:
                
#                 if not(self.zero_T):
#                     self.calc_tunnel_rates()
#                     self.calc_cotunnel_rates()
#                 else:
#                     self.calc_tunnel_rates_zero_T()
#                     self.calc_cotunnel_rates_zero_T()

#                 self.select_co_event(random_number1, random_number2)

#                 if self.co_event_selected:
#                     np1 = self.co_adv_index1[self.jump]
#                     np2 = self.co_adv_index3[self.jump]
#                     self.jump_storage_co[self.jump] += 1
#                 else:
#                     np1 = self.adv_index_rows[self.jump]
#                     np2 = self.adv_index_cols[self.jump]
#                     self.jump_storage[self.jump]    += 1

#             self.charge_mean    += self.charge_vector
#             self.potential_mean += self.potential_vector

#             # If jump from target electrode
#             if (np1 == target_electrode):
#                 self.counter_output_jumps_neg += 1

#             # If jump towards target electrode
#             if (np2 == target_electrode):
#                 self.counter_output_jumps_pos += 1
                        
#             # Statistics
#             self.total_jumps        +=  1
#             new_value               =   (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/self.time
#             delta                   =   new_value - self.jump_diff_mean
#             self.jump_diff_mean     +=  delta/self.total_jumps           
#             delta2                  =   new_value - self.jump_diff_mean
#             self.jump_diff_mean2    +=  delta * delta2

#             # Update relative error and standard deviation if target electrode was involved
#             if ((np1 == target_electrode) or (np2 == target_electrode)):
#                 N_calculations += 1
#                 self.calc_rel_error(N_calculations)
                
#         if (self.total_jumps < 10):
#             self.jump_diff_mean = 0.0

#         self.jump_storage = self.jump_storage/self.time

# def reach_equilibrium(self, min_jumps=1000, max_steps=10, rel_error=0.15, min_nps_eq=0.9, max_jumps=1000000) -> int:
#         """
#         Equilibrate the system

#         Parameters
#         ----------
#         min_jumps : int
#             Minimum amount of jumps in one equilibration step
#         max_steps : int
#             Maximum number of steps before min_jumps is increased
#         rel_error : float
#             Relative error to be achived for changes in nanoparticle potentials
#         min_nps_eq : float
#             Minimum amount of nanoparticles to be equilibrated so that the system is stated as euilibrated 
#         max_jumps : int
#             Maximum jumps in total before function ends
#         """

#         no_equilibrium      = True
#         self.total_jumps    = 0
#         counter             = 0
#         n_trys              = 0
#         N_particles         = len(self.potential_vector) - self.N_electrodes
#         self.time           = 0.0

#         mean_potentials_1   = np.zeros(N_particles)
#         mean_potentials_2   = np.zeros(N_particles)

#         self.calc_potentials()
#         while(no_equilibrium or (self.total_jumps < min_jumps)):

#             if (self.jump == -1):
#                 break
            
#             mean_potentials_1   +=  self.potential_vector[self.N_electrodes:]/max_steps
            
#             random_number1  = np.random.rand()
#             random_number2  = np.random.rand()

#             if self.cotunneling == False:
#                 if not(self.zero_T):
#                     self.calc_tunnel_rates()
#                 else:
#                     self.calc_tunnel_rates_zero_T()
#                 self.select_event(random_number1, random_number2)
#             else:
#                 if not(self.zero_T):
#                     self.calc_tunnel_rates()
#                     self.calc_cotunnel_rates()
#                 else:
#                     self.calc_tunnel_rates_zero_T()
#                     self.calc_cotunnel_rates_zero_T()
#                 self.select_co_event(random_number1, random_number2)
            
#             counter += 1

#             if (counter % max_steps == 0):

#                 self.total_jumps    += max_steps
#                 n_trys              += 1

#                 if (n_trys % 10 == 0):
#                     max_steps = max_steps*2
#                 if (self.total_jumps >= max_jumps):
#                     break

#                 diff = 2*np.abs(np.abs(mean_potentials_1) - np.abs(mean_potentials_2))/(np.abs(mean_potentials_1) + np.abs(mean_potentials_2))
                
#                 if (np.sum(diff < rel_error)/N_particles >= min_nps_eq):
#                     no_equilibrium = False
                    
#                 mean_potentials_2   = mean_potentials_1
#                 mean_potentials_1   = np.zeros(N_particles)

#                 counter = 0
        
#         return self.total_jumps


# def kmc_time_simulation(self, target_electrode : int, time_target : float, store_per_it_min=0, store_per_it_max=0):
#         """
#         Runs KMC until KMC time exceeds a target value

#         Parameters
#         ----------
#         target_electrode : int
#             electrode index of which electric current is estimated
#         time_target : float
#             time value to be reached
#         """


#         # self.counter_output_jumps_neg   = 0
#         # self.counter_output_jumps_pos   = 0
        
#         self.total_jumps    = 0
#         self.charge_mean    = np.zeros(len(self.charge_vector))
#         self.potential_mean = np.zeros(len(self.potential_vector))
#         self.I_network      = np.zeros(len(self.adv_index_rows))


#         # self.jump_dist_per_it           = np.expand_dims(np.zeros(len(self.adv_index_rows)),0)
#         # self.landscape_per_it           = np.expand_dims(np.zeros(len(self.potential_vector)),0)
#         # self.time_vals                  = np.zeros(1)

#         inner_time  = self.time
#         last_time   = 0.0
#         rate_index1 = np.where(self.adv_index_cols == target_electrode)[0][0]
#         rate_index2 = np.where(self.adv_index_rows == target_electrode)[0][0]
#         rate_diffs  = 0.0

#         while (self.time < time_target):

#             # Start time
#             last_time   = self.time         

#             # KMC Part
#             random_number1  = np.random.rand()
#             random_number2  = np.random.rand()

#             # T=0 Approximation of Rates
#             if not(self.zero_T):
#                 self.calc_tunnel_rates()
#             else:
#                 self.calc_tunnel_rates_zero_T()

#             rate1   = self.tunnel_rates[rate_index1]
#             rate2   = self.tunnel_rates[rate_index2]

#             # KMC Step and evolve in time
#             self.select_event(random_number1, random_number2)

#             if (self.jump == -1):
#                 break

#             # Occured jump
#             np1 = self.adv_index_rows[self.jump]
#             np2 = self.adv_index_cols[self.jump]

#             if (self.time >= time_target):
#                 self.neglect_last_event(np1,np2)
#                 break

#             self.I_network[self.jump]    += 1
            
#             self.charge_mean    += self.charge_vector
#             self.potential_mean += self.potential_vector
            
#             # if ((self.time >= store_per_it_min) & (self.time < store_per_it_max)):
#             #     self.jump_dist_per_it   = np.vstack((self.jump_dist_per_it, np.expand_dims(self.I_network,0)))
#             #     self.landscape_per_it   = np.vstack((self.landscape_per_it, np.expand_dims(self.potential_vector,0)))
#             #     self.time_vals          = np.hstack((self.time_vals, np.array([self.time])))

#             # If jump from target electrode
#             if (np1 == target_electrode):
#                 self.counter_output_jumps_neg += 1
#             # If jump towards target electrode
#             if (np2 == target_electrode):
#                 self.counter_output_jumps_pos += 1
            
#             # Statistics
#             self.total_jumps +=  1
        
#         if (self.jump == -1):

#             self.I_target_mean  = 0.0
#             self.I_network      = self.I_network/(time_target-inner_time)

#         if (last_time-inner_time) != 0:
            
#             self.I_target_mean = (self.counter_output_jumps_pos - self.counter_output_jumps_neg)/(time_target-inner_time)
#             self.I_network   = self.I_network/(time_target-inner_time)

#         else:
#             self.I_target_mean = 0