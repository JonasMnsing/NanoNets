# def train_time_series(self, x : np.array, y : np.array, learning_rate : float, batch_size : int, N_epochs : int, epsilon=0.001, adam=False,
#                           time_step=1e-10, stat_size=500, Uc_init=0.1, transient_steps=0, print_nth_batch=1, save_nth_epoch=1, path=''):

#         # Parameter
#         N_voltages          = len(x)
#         N_batches           = int(N_voltages/batch_size)
#         N_controls          = self.N_electrodes - 2
#         time_steps          = time_step*np.arange(N_voltages)
#         control_voltages    = np.random.uniform(-Uc_init, Uc_init, N_controls)
#         target_electrode    = self.N_electrodes - 1
#         y                   = (y - np.mean(y))/np.std(y)

#         # Multiprocessing Manager
#         with multiprocessing.Manager() as manager:

#             # Storage for simulation results
#             return_dic = manager.dict()
#             current_charge_vector = None
            
#             # ADAM Optimization
#             if adam:
#                 m = np.zeros_like(control_voltages)
#                 v = np.zeros_like(control_voltages)

#             for epoch in range(1, N_epochs+1):
                
#                 # Set charge vector to None
#                 predictions = np.zeros(N_voltages)
                
#                 for batch in range(N_batches):

#                     start   = batch*batch_size
#                     stop    = (batch+1)*batch_size
                    
#                     # Voltage array containing input at column 0 
#                     voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
#                     voltages[:,0]       = x[start:stop]
#                     voltages[:,1:-2]    = np.tile(np.round(control_voltages,4), (batch_size,1))
#                     voltages_list       = []
#                     voltages_list.append(voltages)

#                     # Set up a list of voltages considering small deviations
#                     for i in range(N_controls):

#                         voltages_tmp        = voltages.copy()
#                         voltages_tmp[:,i+1] += epsilon
#                         voltages_list.append(voltages_tmp)

#                         voltages_tmp        = voltages.copy()
#                         voltages_tmp[:,i+1] -= epsilon
#                         voltages_list.append(voltages_tmp)

#                     # Container for processes
#                     procs = []

#                     # For each set of voltages assign start a process
#                     for thread in range(2*N_controls+1):

#                         # Make a deep copy of the current instance for each process
#                         instance_copy   = copy.deepcopy(self)

#                         # Start process
#                         process = multiprocessing.Process(target=self.sim_run_for_gradient_decent, args=(thread, instance_copy, return_dic, voltages_list[thread],
#                                                                                                          time_steps[start:stop], target_electrode, stat_size,
#                                                                                                          current_charge_vector))
#                         process.start()
#                         procs.append(process)
                    
#                     # Wait for all processes
#                     for p in procs:
#                         p.join()

#                     # Current charge vector given the last iteration
#                     current_charge_vector = return_dic[-1]
                    
#                     # Gradient Container
#                     gradients = np.zeros_like(control_voltages)

#                     # Calculate gradients for each control voltage 
#                     for i in np.arange(1,2*N_controls+1,2):

#                         y_pred_eps_pos          = return_dic[i]
#                         y_pred_eps_neg          = return_dic[i+1]
#                         y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
#                         y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
#                         perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
#                         perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
#                         gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                    
#                     if ((epoch != 1) and (batch != 0)):
                        
#                         # Current prediction and loss
#                         y_pred  = return_dic[0]
#                         y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
#                         loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                        
#                         predictions[(start+1):stop] = y_pred

#                         # ADAM Optimization
#                         if adam:

#                             beta1 = 0.9         # decay rate for the first moment
#                             beta2 = 0.999       # decay rate for the second moment

#                             # Update biased first and second moment estimate
#                             m = beta1 * m + (1 - beta1) * gradients
#                             v = beta2 * v + (1 - beta2) * (gradients ** 2)

#                             # Compute bias-corrected first and second moment estimate
#                             m_hat = m / (1 - beta1 ** epoch)
#                             v_hat = v / (1 - beta2 ** epoch)

#                             # Update control voltages given the gradients
#                             control_voltages -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

#                         # Update control voltages given the gradients
#                         else:
#                             control_voltages -= learning_rate * gradients

#                         # Print infos
#                         if batch % print_nth_batch == 0:
#                             print(f'Epoch   : {epoch}')
#                             print(f'U_C     : {np.round(control_voltages,4)}')
#                             print(f"Loss    : {loss}")

#                 # Save prediction
#                 if epoch % save_nth_epoch == 0:
#                     np.savetxt(fname=f"{path}ypred_{epoch}.csv", X=predictions)
#                     np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)

#     def train_time_series_by_frequency(self, x : np.array, y : np.array, learning_rate : float, batch_size : int, N_epochs : int, epsilon=0.1, adam=False,
#                           time_step=1e-10, stat_size=500, f_init=1.0, amplitude=0.1, transient_steps=0, print_nth_batch=1, save_nth_epoch=1, path=''):

#         # Parameter
#         N_voltages          = len(x)
#         N_batches           = int(N_voltages/batch_size)
#         N_controls          = self.N_electrodes - 2
#         time_steps          = time_step*np.arange(N_voltages)
#         params              = np.round(np.random.uniform(0.1, f_init, N_controls), 4) #np.repeat(f_init, N_controls)
#         target_electrode    = self.N_electrodes - 1
#         y                   = (y - np.mean(y))/np.std(y)

#         # Multiprocessing Manager
#         with multiprocessing.Manager() as manager:

#             # Storage for simulation results
#             return_dic = manager.dict()
#             current_charge_vector = None
            
#             # ADAM Optimization
#             if adam:
#                 m = np.zeros_like(params)
#                 v = np.zeros_like(params)

#             for epoch in range(1, N_epochs+1):
                
#                 # Set charge vector to None
#                 predictions = np.zeros(N_voltages)
                
#                 for batch in range(N_batches):

#                     start   = batch*batch_size
#                     stop    = (batch+1)*batch_size
                    
#                     # Voltage array containing input at column 0 
#                     voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
#                     voltages[:,0]       = x[start:stop]
#                     for i, f in enumerate(params):
#                         voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
#                     voltages_list       = []
#                     voltages_list.append(voltages)

#                     # Set up a list of voltages considering small deviations
#                     for i in range(N_controls):

#                         voltages_tmp        = voltages.copy()
#                         voltages_tmp[:,i+1] = amplitude*np.cos((f+epsilon)*time_steps[start:stop]*1e8)
#                         voltages_list.append(voltages_tmp)

#                         voltages_tmp        = voltages.copy()
#                         voltages_tmp[:,i+1] = amplitude*np.cos((f-epsilon)*time_steps[start:stop]*1e8)
#                         voltages_list.append(voltages_tmp)

#                     # Container for processes
#                     procs = []

#                     # For each set of voltages assign start a process
#                     for thread in range(2*N_controls+1):

#                         # Make a deep copy of the current instance for each process
#                         instance_copy   = copy.deepcopy(self)

#                         # Start process
#                         process = multiprocessing.Process(target=self.sim_run_for_gradient_decent, args=(thread, instance_copy, return_dic, voltages_list[thread], time_steps,
#                                                                                                     target_electrode, stat_size, current_charge_vector))
#                         process.start()
#                         procs.append(process)
                    
#                     # Wait for all processes
#                     for p in procs:
#                         p.join()

#                     # Current charge vector given the last iteration
#                     current_charge_vector = return_dic[-1]
                    
#                     # Gradient Container
#                     gradients = np.zeros_like(params)

#                     # Calculate gradients for each frequency 
#                     for i in np.arange(1,2*N_controls+1,2):

#                         y_pred_eps_pos          = return_dic[i]
#                         y_pred_eps_neg          = return_dic[i+1]
#                         y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
#                         y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
#                         perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
#                         perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
#                         gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

#                     # Current prediction and loss
#                     y_pred  = return_dic[0]
#                     y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
#                     loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                    
#                     predictions[(start+1):stop] = y_pred

#                     # ADAM Optimization
#                     if adam:

#                         beta1 = 0.9         # decay rate for the first moment
#                         beta2 = 0.999       # decay rate for the second moment

#                         # Update biased first and second moment estimate
#                         m = beta1 * m + (1 - beta1) * gradients
#                         v = beta2 * v + (1 - beta2) * (gradients ** 2)

#                         # Compute bias-corrected first and second moment estimate
#                         m_hat = m / (1 - beta1 ** epoch)
#                         v_hat = v / (1 - beta2 ** epoch)

#                         # Update frequencies given the gradients
#                         params -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

#                     # Update frequencies given the gradients
#                     else:
#                         params -= learning_rate * gradients

#                     params = np.clip(params, 0.1, 10.0)

#                     # Print infos
#                     if batch % print_nth_batch == 0:
#                         print(f'Epoch   : {epoch}')
#                         print(f'U_C     : {np.round(params,4)}')
#                         print(f"Loss    : {loss}")

#                 # Save prediction
#                 if epoch % save_nth_epoch == 0:
#                     np.savetxt(fname=f"{path}ypred_{epoch}.csv", X=predictions)
#                     np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)



# class Optimizer(simulation):

#     def __init__(self, optimizer_type : str, parameter_type: str, topology_parameter : dict, folder='', add_to_path="",
#                  res_info=None, res_info2=None, np_info=None, np_info2=None, seed=None, **kwargs):

#         super().__init__(topology_parameter, folder, add_to_path, res_info, res_info2, np_info, np_info2, seed, **kwargs)
        
#         self.optimizer_type = optimizer_type
#         self.parameter_type = parameter_type
#         self.best_loss      = np.inf
#         self.N_controls     = self.N_electrodes - 2

#     def loss_function(self, y_pred : np.array, y_real : np.array, transient=0)->float:
#         """Error given a norm.

#         Parameters
#         ----------
#         y_pred : np.array
#             Predicted values
#         y_real : np.array
#             Actual values
#         transient : int, optional
#             Neglect the first transient steps, by default 0

#         Returns
#         -------
#         float
#             RMSE
#         """

#         error = np.mean((y_pred[transient:]-y_real[transient:])**2)

#         return error
    
#     def simulate_current_loss(self, voltages : np.array, y_true : np.array, time_steps : np.array, stat_size=500, transient=0, init_charges=None):

#         # Run Simulation
#         target_electrode   = self.N_electrodes - 1
#         self.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
#                               stat_size=stat_size, init_charges=init_charges, save=False)

#         # Extract prediction and current loss
#         output_values   = self.return_output_values()
#         y_pred          = output_values[:,2]
#         self.y_pred     = (y_pred - np.mean(y_pred)) / np.std(y_pred)
#         self.loss       = self.loss_function(y_pred=y_pred, y_real=y_true, transient=transient)
    
#     def simulate_current_loss_parallel(self, thread : int, class_instance, return_dic : dict, voltages : np.array, time_steps : np.array, stat_size=500, init_charges=None):
#         """Simulation execution for gradient decent algorithm. Inits a class and runs simulation for variable voltages.

#         Parameters
#         ----------
#         thread : int
#             Thread if
#         return_dic : dict
#             Dictonary containing simulation results
#         voltages : np.array
#             Voltage values
#         time_steps : np.array
#             Time Steps
#         target_electrode : int
#             Electrode associated to target observable
#         stat_size : int
#             Number of individual runs for statistics
#         network_topology : str
#             Network topology, either 'cubic' or 'random'
#         topology_parameter : dict
#             Dictonary containing information about topology
#         initial_charge_vector : _type_, optional
#             If not None, initial_charge_vector is used as the first network state, by default None
#         """

        
#         target_electrode    = self.N_electrodes - 1
#         class_instance.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode,
#                                         stat_size=stat_size, init_charges=init_charges, save=False)
            
#         # Get target observable 
#         output_values       = class_instance.return_output_values()
#         observable          = output_values[:,2]
#         error_values        = output_values[:,3]
#         return_dic[thread]  = observable

#         if thread == 0:
#             charge_values   = class_instance.q_eq
#             return_dic[-1]  = charge_values

#     def simulated_annealing(self, x : np.array, y : np.array, N_epochs : int, temp_init : float, cooling_rate=0.9, epsilon=0.01,
#                             time_step=1e-10, stat_size=500, p_init=0.1, amplitude=0.02, freq=2.0, batch_size=0, print_nth_epoch=1, save_nth_epoch=1):

#         # Target Values
#         y           = (y - np.mean(y))/np.std(y)
#         charge_init = None

#         # Initial Values
#         N_voltages  = len(x)

#         if batch_size == 0:
#             N_batches   = 1
#             batch_size  = N_voltages
#         else:
#             N_batches   = N_voltages//batch_size

#         if self.parameter_type == 'freq':
#             self.best_params    = np.random.uniform(0.1, p_init, self.N_controls)
#         elif self.parameter_type == 'phase':
#             self.best_params    = np.random(0.0, 2*np.pi, self.N_controls)
#         else:
#             self.best_params    = np.random.uniform(-p_init, p_init, self.N_controls)

#         time_steps  = time_step*np.arange(N_voltages)
#         temp        = temp_init

#         # For each epoch
#         for epoch in range(1, N_epochs+1):

#             for batch in range(N_batches):

#                 start   = batch*batch_size
#                 stop    = (batch+1)*batch_size
            
#                 # New parameter values
#                 params              = self.best_params + np.random.uniform(-epsilon, epsilon, self.N_controls)
#                 voltages            = np.zeros(shape=(batch_size, self.N_electrodes+1))
#                 voltages[:,0]       = x[start:stop]

#                 if self.parameter_type == 'freq':
#                     params = np.clip(params, a_min=0.1, a_max=10.0)
#                     for i, f in enumerate(params):
#                         voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
#                 elif self.parameter_type == 'phase':
#                     for i, phi in enumerate(params):
#                         voltages[:,i+1] = amplitude*np.cos(freq*time_steps[start:stop]*1e8-phi)
#                 else:
#                     voltages[:,1:-2]    = np.tile(np.round(params,4), (batch_size,1))
            
#                 # Run Simulation and get new loss
#                 self.simulate_current_loss(voltages=voltages, y_true=y[start+1:stop], time_steps=time_steps[start:stop], stat_size=stat_size, transient=1000, init_charges=charge_init)
            
#                 delta_loss  = self.loss - self.best_loss
#                 charge_init = self.q_eq

#                 # Check for new best parameters
#                 if delta_loss < 0:
#                     self.best_loss      = self.loss
#                     self.best_params    = params
#                 else:
#                     prob = min([1.0, np.exp(-delta_loss/temp)])
#                     if np.random.rand() < prob:
#                         self.best_loss      = self.loss
#                         self.best_params    = params

#                 # Cool down temperature
#                 if cooling_rate == 0:
#                     temp = temp/np.log(1+epoch)
#                 else:
#                     temp = temp * cooling_rate

#             # Print infos
#             if epoch % print_nth_epoch == 0:
#                 print(f'Epoch   : {epoch}')
#                 print(f'U_C     : {np.round(self.best_params,4)}')
#                 print(f"Loss    : {self.best_loss}")
#                 print(f"Temp    : {temp}")

#             # Save prediction
#             if epoch % save_nth_epoch == 0:
#                 np.savetxt(fname=f"{self.folder}ypred_{epoch}.csv", X=self.y_pred)

#     def gradient_decent(self, x : np.array, y : np.array, N_epochs : int, learning_rate : float, batch_size : int, epsilon=0.001, adam=False,
#                         time_step=1e-10, stat_size=500, p_init=0.1, amplitude=0.02, transient_steps=0, print_nth_batch=1, save_nth_epoch=1):
        
#         # Target Values
#         y   = (y - np.mean(y))/np.std(y)

#         # Initial Values
#         N_voltages  = len(x)
#         N_batches   = N_voltages//batch_size
#         time_steps  = time_step*np.arange(N_voltages)

#         if self.parameter_type == 'freq':
#             params  = np.random.uniform(0.1, p_init, self.N_controls)
#         else:
#             params  = np.random.uniform(-p_init, p_init, self.N_controls)

#         # Multiprocessing Manager
#         with multiprocessing.Manager() as manager:

#             # Storage for simulation results
#             return_dic  = manager.dict()
#             charge_init = None
            
#             # ADAM Optimization
#             if adam:
#                 m = np.zeros_like(params)
#                 v = np.zeros_like(params)

#             for epoch in range(1, N_epochs+1):
                
#                 # Set charge vector to None
#                 predictions = np.zeros(N_voltages)
                
#                 for batch in range(N_batches):

#                     start   = batch*batch_size
#                     stop    = (batch+1)*batch_size
                    
#                     # Voltage array containing input at column 0 
#                     voltages        = np.zeros(shape=(batch_size, self.N_electrodes+1))
#                     voltages[:,0]   = x[start:stop]
                    
#                     if self.parameter_type == 'freq':
#                         for i,f in enumerate(params):
#                             voltages[:,i+1] = amplitude*np.cos(np.round(f,4)*time_steps[start:stop]*1e8)
#                         voltages_list       = []
#                         voltages_list.append(voltages)

#                         # Set up a list of voltages considering small deviations
#                         for i,f in enumerate(params):

#                             voltages_tmp        = voltages.copy()
#                             voltages_tmp[:,i+1] = amplitude*np.cos((f+epsilon)*time_steps[start:stop]*1e8)
#                             voltages_list.append(voltages_tmp)

#                             voltages_tmp        = voltages.copy()
#                             voltages_tmp[:,i+1] = amplitude*np.cos((f-epsilon)*time_steps[start:stop]*1e8)
#                             voltages_list.append(voltages_tmp)
#                     else:
#                         voltages[:,1:-2]    = np.tile(np.round(params,4), (batch_size,1))
#                         voltages_list       = []
#                         voltages_list.append(voltages)

#                         # Set up a list of voltages considering small deviations
#                         for i in range(self.N_controls):

#                             voltages_tmp        = voltages.copy()
#                             voltages_tmp[:,i+1] += epsilon
#                             voltages_list.append(voltages_tmp)

#                             voltages_tmp        = voltages.copy()
#                             voltages_tmp[:,i+1] -= epsilon
#                             voltages_list.append(voltages_tmp)

#                     # Container for processes
#                     procs       = []

#                     # For each set of voltages assign start a process
#                     for thread in range(2*self.N_controls+1):
                        
#                         self_copy = copy.deepcopy(self)

#                         # Start process
#                         process = multiprocessing.Process(target=self.simulate_current_loss_parallel, args=(thread, self_copy, return_dic, voltages,
#                                                                                                             time_steps, stat_size, charge_init))
#                         process.start()
#                         procs.append(process)
                    
#                     # Wait for all processes
#                     for p in procs:
#                         p.join()

#                     # Current charge vector given the last iteration
#                     charge_init = return_dic[-1]
                    
#                     # Gradient Container
#                     gradients = np.zeros_like(params)

#                     # Calculate gradients for each control voltage 
#                     for i in np.arange(1,2*self.N_controls+1,2):

#                         y_pred_eps_pos          = return_dic[i]
#                         y_pred_eps_neg          = return_dic[i+1]
#                         y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
#                         y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
#                         perturbed_loss_pos      = self.loss_function(y_pred=y_pred_eps_pos, y_real=y[(start+1):stop], transient=transient_steps)
#                         perturbed_loss_neg      = self.loss_function(y_pred=y_pred_eps_neg, y_real=y[(start+1):stop], transient=transient_steps)
#                         gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

                    
#                     # if ((epoch != 1) and (batch != 0)):
                        
#                     # Current prediction and loss
#                     y_pred  = return_dic[0]
#                     y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
#                     loss    = self.loss_function(y_pred=y_pred, y_real=y[(start+1):stop], transient=transient_steps)
                    
#                     predictions[(start+1):stop] = y_pred

#                     # ADAM Optimization
#                     if adam:

#                         beta1 = 0.9         # decay rate for the first moment
#                         beta2 = 0.999       # decay rate for the second moment

#                         # Update biased first and second moment estimate
#                         m = beta1 * m + (1 - beta1) * gradients
#                         v = beta2 * v + (1 - beta2) * (gradients ** 2)

#                         # Compute bias-corrected first and second moment estimate
#                         m_hat = m / (1 - beta1 ** epoch)
#                         v_hat = v / (1 - beta2 ** epoch)

#                         # Update control voltages given the gradients
#                         params -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

#                     # Update control voltages given the gradients
#                     else:
#                         params -= learning_rate * gradients

#                     # Print infos
#                     if batch % print_nth_batch == 0:
#                         print(f'Epoch   : {epoch}')
#                         print(f'U_C     : {np.round(params,4)}')
#                         print(f"Loss    : {loss}")

#                 # Save prediction
#                 if epoch % save_nth_epoch == 0:
#                     np.savetxt(fname=f"{self.folder}ypred_{epoch}.csv", X=predictions)
#                     # np.savetxt(fname=f"{path}charge_{epoch}.csv", X=current_charge_vector)


# def kmc_time_simulation_potential2(self, target_electrode : int, time_target : float):
#         """
#         Runs KMC until KMC time exceeds a target value

#         Parameters
#         ----------
#         target_electrode : int
#             electrode index of which electric current is estimated
#         time_target : float
#             time value to be reached
#         """

#         # Calculate potential landscape once
#         self.calc_potentials()
        
#         self.total_jumps    = 0
#         self.charge_mean    = np.zeros(len(self.charge_vector))
#         self.potential_mean = np.zeros(len(self.potential_vector))
#         self.I_network      = np.zeros(len(self.adv_index_rows))
#         idx_np_target       = self.adv_index_cols[self.floating_electrodes]

#         inner_time          = self.time
#         last_time           = 0.0        
#         target_potential    = 0.0

#         while (self.time < time_target):

#             # KMC Part
#             random_number1  = np.random.rand()
#             random_number2  = np.random.rand()

#             # T=0 Approximation of Rates
#             if not(self.zero_T):
#                 self.calc_tunnel_rates()
#             else:
#                 self.calc_tunnel_rates_zero_T()

#             # KMC Step and evolve in time
#             self.select_event(random_number1, random_number2)

#             if (self.jump == -1):
#                 break

#             # Occured jump
#             np1 = self.adv_index_rows[self.jump]
#             np2 = self.adv_index_cols[self.jump]

#             # If time exceeds target time
#             if (self.time >= time_target):
#                 self.neglect_last_event(np1,np2)
#                 break

#             # Update potential of floating target electrode
#             self.update_floating_electrode(idx_np_target)
#             target_potential += self.potential_vector[target_electrode]*(self.time-last_time)

#             # Update Observables
#             self.charge_mean            += self.charge_vector*(self.time-last_time)
#             self.potential_mean         += self.potential_vector*(self.time-last_time)
#             self.I_network[self.jump]   += 1            
#             self.total_jumps            += 1           
        
#         if (self.jump == -1):
#             self.target_observable_mean  = self.potential_vector[target_electrode]

#         if (last_time-inner_time) != 0:
#             self.target_observable_mean = target_potential/(time_target-inner_time)
#             self.I_network              = self.I_network/self.total_jumps
#             self.charge_mean            = self.charge_mean/(time_target-inner_time)
#             self.potential_mean         = self.potential_mean/(time_target-inner_time)
            
#         else:
#             self.target_observable_mean = self.potential_vector[target_electrode]
#             self.I_network              = self.I_network
#             self.charge_mean            = self.charge_vector
#             self.potential_mean         = self.potential_vector