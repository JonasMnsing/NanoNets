import logging
import multiprocessing
from pathlib import Path
import numpy as np
import nanonets

def run_simulation(time_steps, voltages, topology, out_folder, stat_size, T_val):
    """Instantiate and run one sim at temperature T_val."""
    try:
        target = len(topology["e_pos"]) - 1
        sim = nanonets.simulation(
            topology_parameter=topology,
            folder=str(out_folder),
            high_C_output=False
        )
        sim.run_var_voltages(
            voltages=voltages,
            time_steps=time_steps,
            target_electrode=target,
            stat_size=stat_size,
            save=True,
            T_val=T_val
        )
        logging.info(f"Done N={topology['Nx']} @ T={T_val}K")
    except Exception:
        logging.exception(f"Error for topology {topology} @ T={T_val}K")

def batch_launch(func, tasks, max_procs):
    running = []
    for args in tasks:
        while len(running) >= max_procs:
            running.pop(0).join()
        p = multiprocessing.Process(target=func, args=args)
        p.start()
        running.append(p)
    for p in running:
        p.join()

def build_voltage_matrix(n_steps, n_elec, V_drive):
    volt        = np.zeros((n_steps, n_elec+1), float)
    volt[:, 0]  = V_drive
    return volt