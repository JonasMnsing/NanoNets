import logging
import numpy as np
from pathlib import Path
from nanonets.utils import sinusoidal_voltages, batch_launch, run_dynamic_simulation

# ─── Configuration ───
N_PARTICLES     = 9
N_ELECTRODES    = 8
AMPLITUDE       = [0.02] + [0.0]*(N_ELECTRODES-1)
FREQUENCY       = [16.0 * 1e6] + [0.0]*(N_ELECTRODES-1)
V_GATE          = 0.0
N_PERIODS       = 100
N_PROCS         = 1
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/nanonets/scripts/2_funding_period/WP2/")
STAT_SIZE       = 200
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    topo = {"Nx": N_PARTICLES, "Ny": N_PARTICLES,
            "e_pos" : [[int((N_PARTICLES-1)/2),0], [0,0], [N_PARTICLES-1,0],
                       [0,int((N_PARTICLES-1)/2)], [N_PARTICLES-1,int((N_PARTICLES)/2)], 
                       [0,N_PARTICLES-1], [N_PARTICLES-1,N_PARTICLES-1],
                       [int((N_PARTICLES)/2),(N_PARTICLES-1)]],
            "electrode_type" : ['constant']*(N_ELECTRODES-1) + ['floating']}
    dt          = 1/(40 * FREQUENCY[0])
    T_sim       = N_PERIODS / FREQUENCY[0]
    N_steps     = int(np.ceil(T_sim / dt))
    time, volt  = sinusoidal_voltages(N_steps, topo, AMPLITUDE, FREQUENCY, time_step=dt)
    
    args    = (time, volt, topo,PATH)
    kwargs  = {"net_kwargs":{"pack_optimizer":False},
                "sim_kwargs":{'stat_size':STAT_SIZE, 'save':True}}
    tasks.append((args,kwargs))

    batch_launch(run_dynamic_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()