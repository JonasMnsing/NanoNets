import multiprocessing
import numpy as np
import sys

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets

def estimate_z(voltages, time_steps, network_topology, topology_parameter, stat_size, eq_steps):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, save=False)
    output_values = sim_class.return_output_values()

    return output_values[:,2]

def estimate_z_parallel(thread, return_dic, voltages, time_steps, network_topology, topology_parameter, stat_size, eq_steps, np_info):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter, np_info=np_info, seed=4)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, save=False, output_potential=True)
    output_values = sim_class.return_output_values()

    return_dic[thread]  = output_values[:,2]

def loss_function(z_pred, z_real, transient=1000):
    
    return np.mean((z_pred[transient:]-z_real[transient:])**2)

def return_voltage_array(x_vals, control_voltages, N_voltages, topology_parameter):

    voltages            = np.zeros(shape=(N_voltages, len(topology_parameter["e_pos"])+1))
    voltages[:,0]       = x_vals
    voltages[:,1:-2]    = np.tile(control_voltages, (N_voltages,1))
    
    return voltages

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
    "std_radius"    : 2.0,  # standard deviation of nanoparticle radius
    "np_distance"   : 1.0   # spacing between nanoparticle shells
}

# x_vals and z_vals
max_control = 0.05
std_input   = 0.05
x_vals      = np.loadtxt(f"{folder}x_vals.csv")
z_vals      = np.loadtxt(f"{folder}z_vals.csv")
x_vals      = std_input*(x_vals - np.mean(x_vals)) / np.std(x_vals)
z_vals      = (z_vals - np.mean(z_vals)) / np.std(z_vals)

# Time Vals
step_size   = 1e-10
N_voltages  = len(x_vals)
N_controls  = len(topology_parameter["e_pos"]) - 2
time_steps  = step_size*np.arange(N_voltages)

# Initial control voltages
control_voltages = np.random.uniform(-max_control, max_control, int(N_controls))

# Multiprocessing Manager
manager     = multiprocessing.Manager()
return_dic  = manager.dict()

# Optimization
for epoch in range(N_epochs):

    print(f'Run : {epoch}')
    print(f'U_C : {control_voltages}')
    
    # For current set of controls run simulation and estimate error
    voltages        = return_voltage_array(x_vals, control_voltages, N_voltages, topology_parameter)
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

        process = multiprocessing.Process(target=estimate_z_parallel, args=(thread, return_dic, voltages_list[thread], time_steps,
                                                                            network_topology, topology_parameter, stat_size, eq_steps, np_info))
        process.start()
        procs.append(process)
    
    # Wait for all threads
    for p in procs:
        p.join()

    # Gradient Container
    gradients = np.zeros_like(control_voltages)

    # Calculate gradients
    for i in np.arange(1,2*N_controls+1,2):

        z_pred_eps_pos          = return_dic[i]
        z_pred_eps_neg          = return_dic[i+1]
        z_pred_eps_pos          = (z_pred_eps_pos - np.mean(z_pred_eps_pos)) / np.std(z_pred_eps_pos)
        z_pred_eps_neg          = (z_pred_eps_neg - np.mean(z_pred_eps_neg)) / np.std(z_pred_eps_neg)
        perturbed_loss_pos      = loss_function(z_pred=z_pred_eps_pos, z_real=z_vals[1:], transient=transient_steps)
        perturbed_loss_neg      = loss_function(z_pred=z_pred_eps_neg, z_real=z_vals[1:], transient=transient_steps)
        gradients[int((i-1)/2)] = (perturbed_loss_pos - perturbed_loss_neg) / (2*epsilon)

    # Current prediction and loss
    z_pred  = return_dic[0]
    z_pred  = (z_pred - np.mean(z_pred)) / np.std(z_pred)
    loss    = loss_function(z_pred=z_pred, z_real=z_vals[1:], transient=transient_steps)

    # Now update control voltages given the gradients
    control_voltages -= learning_rate * gradients

    print(f"Loss : {loss}")

    # Save prediction
    if epoch % save_nth_epoch == 0:
        np.savetxt(fname=f"{save_folder}z_pred_{epoch}.csv", X=z_pred)
