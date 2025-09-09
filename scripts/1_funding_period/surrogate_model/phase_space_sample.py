import logging
import numpy as np
from pathlib import Path
from nanonets.utils import distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_PARTICLES = 9
N_E         = 8
V_CONTROL   = 0.1
V_GATE      = 0.0
N_DATA      = 512000
N_PROCS     = 64
LOG_LEVEL   = logging.INFO
PATH        = Path("/scratch/j_mens07/data/1_funding_period/phase_space_sample/")
SAVE_TH     = 10
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    topo = {"Nx": N_PARTICLES, "Ny": N_PARTICLES,
            "e_pos" : [[0,0], [int((N_PARTICLES-1)/2),0], [N_PARTICLES-1,0],
                       [0,int((N_PARTICLES-1)/2)], [0,N_PARTICLES-1],
                       [N_PARTICLES-1,int((N_PARTICLES)/2)],
                       [int((N_PARTICLES)/2),(N_PARTICLES-1)], [N_PARTICLES-1,N_PARTICLES-1]],
            "electrode_type" : ['constant']*N_E}
    volt = np.random.uniform(-V_CONTROL, V_CONTROL, size=(N_DATA,N_E-1))
    volt = np.hstack((volt,np.zeros(shape=(N_DATA,2))))
    
    for p in range(N_PROCS):
        volt_p  = distribute_array_across_processes(p, volt, N_PROCS)
        args    = (volt_p, topo, PATH)
        kwargs  = {"net_kwargs":{"pack_optimizer":False},
                    "sim_kwargs":{"save_th":SAVE_TH}}
        tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()