from copy import deepcopy
import multiprocessing
import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets

def sim_run_for_gradient_decent(thread, sim_class, voltages, time_steps, target_electrode, return_dic, stat_size):

    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, stat_size=stat_size, save=False, output_potential=True)
    output_values = sim_class.return_output_values()

    return_dic[thread] = output_values[:,2]

def loss_function(y_pred, y_real, transient=1000):
    
    return np.mean((y_pred[transient:]-y_real[transient:])**2)

def time_series_gradient_decent(sim_class : nanonets.simulation, x_vals : np.array, y_target : np.array, learning_rate : float, N_epochs : int, 
                                epsilon=0.001, time_step=1e-10, stat_size=500, Uc_init=0.05, transient_steps=1000, print_info=False):
    
    N_voltages          = len(x_vals)
    N_controls          = sim_class.N_electrodes - 2
    time_steps          = time_step*np.arange(N_voltages)
    control_voltages    = np.random.uniform(-Uc_init, Uc_init, N_controls)
    target_electrode    = sim_class.N_electrodes - 1
    
    # Multiprocessing Manager
    manager     = multiprocessing.Manager()
    return_dic  = manager.dict()

    for epoch in range(N_epochs):

        voltages            = np.zeros(shape=(N_voltages, sim_class.N_electrodes+1))
        voltages[:,0]       = x_vals
        voltages[:,1:-2]    = np.tile(control_voltages, (N_voltages,1))

        voltages_list   = []
        voltages_list.append(voltages)

        # Append to list
        for i in range(N_controls):

            voltages_tmp        = voltages.copy()
            voltages_tmp[:,i+1] += epsilon
            voltages_list.append(voltages_tmp)

            voltages_tmp        = voltages.copy()
            voltages_tmp[:,i+1] -= epsilon
            voltages_list.append(voltages_tmp)

        # Container for processes
        procs   = []

        # Run parallel simulation
        for thread in range(2*N_controls+1):

            class_copy  = deepcopy(sim_class)

            process = multiprocessing.Process(target=sim_run_for_gradient_decent, args=(thread, class_copy, voltages_list[thread], time_steps,
                                                                                        target_electrode, return_dic, stat_size))
            process.start()
            procs.append(process)
        
        # Wait for all threads
        for p in procs:
            p.join()

        # Gradient Container
        gradients = np.zeros_like(control_voltages)

        # Calculate gradients
        for i in np.arange(1,2*N_controls+1,2):

            y_pred_eps_pos          = return_dic[i]
            y_pred_eps_neg          = return_dic[i+1]
            y_pred_eps_pos          = (y_pred_eps_pos - np.mean(y_pred_eps_pos)) / np.std(y_pred_eps_pos)
            y_pred_eps_neg          = (y_pred_eps_neg - np.mean(y_pred_eps_neg)) / np.std(y_pred_eps_neg)
            perturbed_loss_pos      = loss_function(y_pred=y_pred_eps_pos, y_real=y_target[1:], transient=transient_steps)
            perturbed_loss_neg      = loss_function(y_pred=y_pred_eps_neg, y_real=y_target[1:], transient=transient_steps)
            gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

        # Current prediction and loss
        y_pred  = return_dic[0]
        y_pred  = (y_pred - np.mean(y_pred)) / np.std(y_pred)
        loss    = loss_function(y_pred=y_pred, y_real=y_target[1:], transient=transient_steps)

        # Now update control voltages given the gradients
        control_voltages -= learning_rate * gradients

        if print_info:
            print(f'Run : {epoch}')
            print(f'U_C : {control_voltages}')
            print(f"Loss : {loss}")

        # Save prediction
        # if epoch % save_nth_epoch == 0:
        #     np.savetxt(fname=f"{save_folder}z_pred_{epoch}.csv", X=z_pred)

# Parameter
N_epochs            = 1000
transient_steps     = 4000
stat_size           = 20
eq_steps            = 0
folder              = "scripts/2_funding_period/WP2/lorenz_system/"
save_folder         = "scripts/2_funding_period/WP2/lorenz_system/data/"
network_topology    = 'cubic'
epsilon             = 0.001
learning_rate       = 0.00001
save_nth_epoch      = 1

# Network Topology
topology_parameter  = {
    "Nx"                : 7,
    "Ny"                : 7,
    "Nz"                : 1,
    "e_pos"             : [[0,0,0],[3,0,0],[0,3,0],[6,0,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]],
    "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
}
np_info = {
    "eps_r"         : 2.6,  # Permittivity of molecular junction 
    "eps_s"         : 3.9,  # Permittivity of oxide layer
    "mean_radius"   : 10.0, # average nanoparticle radius
    "std_radius"    : 0.0,  # standard deviation of nanoparticle radius
    "np_distance"   : 1.0   # spacing between nanoparticle shells
}

N_voltages  = 10000
x_vals      = np.random.normal(loc=0.0, scale=0.05, size=N_voltages)
shift       = 1
y_target    = x_vals[:-shift]
x_vals      = x_vals[shift:]

sim_class = nanonets.simulation(network_topology='cubic', topology_parameter=topology_parameter, folder='', np_info=np_info)
time_series_gradient_decent(sim_class=sim_class, x_vals=x_vals, y_target=y_target,
                            learning_rate=learning_rate, epochs=N_epochs, epsilon=epsilon)
