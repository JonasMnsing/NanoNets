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

def estimate_z_parallel(thread, return_dic, voltages, time_steps, network_topology, topology_parameter, stat_size, eq_steps):

    target_electrode = len(topology_parameter["e_pos"]) - 1
    
    sim_class = nanonets.simulation(network_topology=network_topology, topology_parameter=topology_parameter)
    sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=target_electrode, eq_steps=eq_steps, stat_size=stat_size, save=False)
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
stat_size           = 1000
eq_steps            = 0
folder              = "scripts/2_funding_period/WP2/lorenz_system/"
network_topology    = 'cubic'
epsilon             = 0.001
learning_rate       = 0.01

# Network Topology
topology_parameter  = {
    "Nx"    : 7,
    "Ny"    : 7,
    "Nz"    : 1,
    "e_pos" : [[0,0,0],[3,0,0],[0,3,0],[6,0,0],[0,6,0],[6,3,0],[3,6,0],[6,6,0]]
}

# x_vals and z_vals
max_input   = 0.2
max_control = 0.2
x_vals      = np.loadtxt(f"{folder}x_vals.csv")
z_vals      = np.loadtxt(f"{folder}z_vals.csv")
x_vals      = 2*max_input*(x_vals - np.min(x_vals))/(np.max(x_vals) - np.min(x_vals)) - max_input
z_vals      = 2*max_input*(z_vals - np.min(z_vals))/(np.max(z_vals) - np.min(z_vals)) - max_input

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

    for i in range(N_controls):
        voltages_tmp        = voltages.copy()
        voltages_tmp[:,i+1] += epsilon
        voltages_list.append(voltages_tmp)
    
    # Container for processes and results
    # manager     = multiprocessing.Manager()
    # return_dic  = manager.dict()
    procs       = []

    for thread in range(N_controls+1):

        process = multiprocessing.Process(target=estimate_z_parallel, args=(thread, return_dic, voltages_list[thread], time_steps, network_topology, topology_parameter, stat_size, eq_steps))
        process.start()
        procs.append(process)
    
    for p in procs:
        p.join()

    # Current prediction and loss
    z_pred          = return_dic[0]
    gradients       = np.zeros_like(control_voltages)
    current_loss    = loss_function(z_pred=z_pred, z_real=z_vals[1:], transient=transient_steps)

    # Calculate gradients
    for i in range(1, N_controls+1):

        z_pred_eps      = return_dic[i]
        perturbed_loss  = loss_function(z_pred=z_pred_eps, z_real=z_vals[1:], transient=transient_steps)
        gradients[i-1]  = (perturbed_loss - current_loss) / epsilon

    # Now update control voltages given the gradients
    control_voltages -= learning_rate * gradients

    print(f'Error : {current_loss}\n')