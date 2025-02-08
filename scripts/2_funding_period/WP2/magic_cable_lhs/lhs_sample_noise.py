import numpy as np
import sys
import multiprocessing

# Add to path
sys.path.append("/home/jonas/phd/NanoNets/src/")
sys.path.append("/home/j/j_mens07/NanoNets/src/")
sys.path.append("/mnt/c/Users/jonas/Desktop/phd/NanoNets/src/")
sys.path.append("/home/jonasmensing/phd/NanoNets/src/")

# NanoNets Simulation Tool
import nanonets
import nanonets_utils

def return_lhs_sample(pmin, pmax, N_controls, N_samples):

    p_range     = [[pmin,pmax] for _ in range(N_controls)]
    lhs_sample = np.zeros((N_samples, N_controls))
    for i in range(N_controls):
        intervals   = np.linspace(0, 1, N_samples + 1)
        points      = np.random.uniform(intervals[:-1], intervals[1:])
        np.random.shuffle(points)
        lhs_sample[:, i] = points
    lhs_rescale = np.zeros_like(lhs_sample)

    for i in range(N_controls):
        lhs_rescale[:,i] = p_range[i][0] + lhs_sample[:,i] * (p_range[i][1] - p_range[i][0])

    return lhs_rescale

def run_sim(thread, params, rows, time_step, topology, path, sim_type, amplitude, freq, N_voltages, stat_size):

    np_info2 = {
        'np_index'      : [int(topology["Nx"]**2-(topology["Nx"]+1)//2)], 
        'mean_radius'   : 1e6,
        'std_radius'    : 0.0
    }

    thread_rows     = rows[thread]
    params_new      = params[thread_rows,:]
    rs              = np.random.RandomState(seed=0)

    if sim_type == 'offset':
        for n, par in enumerate(params_new):
            
            time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter=topology, amplitudes=amplitude,
                                                                         frequencies=freq*1e5, offset=par, time_step=time_step)
            voltages[:,0]           = rs.uniform(-amplitude,amplitude,len(time_steps))
            sim_class               = nanonets.simulation(topology_parameter=topology, folder=path, seed=0, add_to_path=f'_{thread}_{n}', np_info2=np_info2)
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)

    elif sim_type == 'amplitude':
        for n, par in enumerate(params_new):

            time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter=topology, amplitudes=par,
                                                                        frequencies=freq*1e5, time_step=time_step)
            voltages[:,0]           = rs.uniform(-amplitude,amplitude,len(time_steps))
            sim_class               = nanonets.simulation(topology_parameter=topology, folder=path, seed=0,
                                                          add_to_path=f'_{thread}_{n}', np_info2=np_info2)
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)

    elif sim_type == 'frequency':
        for n, par in enumerate(params_new):
           
            time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter=topology, amplitudes=amplitude,
                                                                        frequencies=par*1e5, time_step=time_step)
            voltages[:,0]           = rs.uniform(-amplitude,amplitude,len(time_steps))
            sim_class               = nanonets.simulation(topology_parameter=topology, folder=path, seed=0,
                                                          add_to_path=f'_{thread}_{n}', np_info2=np_info2)
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)

    elif sim_type == 'phase':
        for n, par in enumerate(params_new):

            time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter=topology, amplitudes=amplitude,
                                                                        frequencies=freq*1e5, phase=par, time_step=time_step)
            voltages[:,0]           = rs.uniform(-amplitude,amplitude,len(time_steps))
            sim_class               = nanonets.simulation(topology_parameter=topology, folder=path, seed=0,
                                                          add_to_path=f'_{thread}_{n}', np_info2=np_info2)
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)

    else:
        for n, par in enumerate(params_new):
            
            time_steps, voltages    = nanonets_utils.sinusoidal_voltages(N_voltages, topology_parameter=topology, amplitudes=amplitude,
                                                                        frequencies=0.0, offset=par, time_step=time_step)
            voltages[:,0]           = rs.uniform(-amplitude,amplitude,len(time_steps))
            sim_class               = nanonets.simulation(topology_parameter=topology, folder=path, seed=0,
                                                          add_to_path=f'_{thread}_{n}', np_info2=np_info2)
            sim_class.run_var_voltages(voltages=voltages, time_steps=time_steps, target_electrode=7, stat_size=stat_size)

if __name__ == '__main__':

    # Network Topology
    N_p                 = 11
    topology_parameter  = {
        "Nx"                : N_p,
        "Ny"                : N_p,
        "Nz"                : 1,
        "e_pos"             : [[(N_p-1)//2,0,0],[0,0,0],[N_p-1,0,0],
                            [0,(N_p-1)//2,0],[N_p-1,(N_p-1)//2,0],[0,N_p-1,0],
                            [N_p-1,N_p-1,0],[(N_p-1)//2,N_p-1,0]],
        "electrode_type"    : ['constant','constant','constant','constant','constant','constant','constant','floating']
    }

    # Parameter
    N_voltages  = 10000
    N_samples   = 512
    stat_size   = 10
    N_procs     = 36
    amplitude   = 0.1
    freq        = 1.0
    time_step   = 1e-7
    N_electrode = len(topology_parameter["e_pos"])
    index       = [i for i in range(N_samples)]
    rows        = [index[i::N_procs] for i in range(N_procs)]
    path        = "/scratch/tmp/j_mens07/data/2_funding_period/potential/magic_cable/noise/"
    
    # Const
    sim_type    = 'const'
    sample      = return_lhs_sample(-0.1, 0.1, N_electrode, N_samples)
    procs       = []
    for i in range(N_procs):
        
        process = multiprocessing.Process(target=run_sim, args=(i, sample, rows, time_step, topology_parameter, path+f'{sim_type}/',
                                                                sim_type, amplitude, freq, N_voltages, stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()
    
    # Amplitude
    sim_type    = 'amplitude'
    sample      = return_lhs_sample(0.0, 0.1, N_electrode, N_samples)
    procs       = []
    for i in range(N_procs):
        
        process = multiprocessing.Process(target=run_sim, args=(i, sample, rows, time_step, topology_parameter, path+f'{sim_type}/',
                                                                sim_type, amplitude, freq, N_voltages, stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()
    
    # Phase
    sim_type    = 'phase'
    sample      = return_lhs_sample(0.0, 2*np.pi, N_electrode, N_samples)
    procs       = []
    for i in range(N_procs):
        
        process = multiprocessing.Process(target=run_sim, args=(i, sample, rows, time_step, topology_parameter, path+f'{sim_type}/',
                                                                sim_type, amplitude, freq, N_voltages, stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # Offset
    sim_type    = 'offset'
    sample      = return_lhs_sample(-0.1, 0.1, N_electrode, N_samples)
    procs       = []
    for i in range(N_procs):
        
        process = multiprocessing.Process(target=run_sim, args=(i, sample, rows, time_step, topology_parameter, path+f'{sim_type}/',
                                                                sim_type, amplitude, freq, N_voltages, stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()

    # Frequency
    sim_type    = 'frequency'
    sample      = return_lhs_sample(0.5, 2.0, N_electrode, N_samples)
    procs       = []
    for i in range(N_procs):
        
        process = multiprocessing.Process(target=run_sim, args=(i, sample, rows, time_step, topology_parameter, path+f'{sim_type}/',
                                                                sim_type, amplitude, freq, N_voltages, stat_size))
        process.start()
        procs.append(process)
    for p in procs:
        p.join()