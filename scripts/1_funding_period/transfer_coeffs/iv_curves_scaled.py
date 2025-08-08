import logging
import numpy as np
from pathlib import Path

# NanoNets
from nanonets.utils import run_static_simulation, batch_launch

# ─── Configuration ───
N_MIN, N_MAX    = 3, 16
N_E             = 8
V_MIN, V_MAX    = 0.0, 1.0
N_DATA          = 100
N_PROCS         = 10
T_VAL           = 5
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/electrode_impact/voltage_sweep/")
VOLTAGE_RANGE   = np.load("scripts/1_funding_period/electrode_impact/voltage_range.npy")
INPUT_POS       = [0,1,2,5]
# ---------------------

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for i, n in enumerate(range(N_MIN, N_MAX+1)):
        topo = {"Nx": n, "Ny": n,
                "e_pos" : [[0,0], 
                           [int((n-1)/2),0],
                           [n-1,0], 
                           [0,int((n-1)/2)],
                           [0,n-1],
                           [n-1,int((n)/2)],
                           [int((n)/2),(n-1)],
                           [n-1,n-1]],
                "electrode_type" : ['constant']*N_E}
        for pos in INPUT_POS:
            volt        = np.zeros(shape=(N_DATA,N_E+1))
            volt[:,pos] = np.linspace(V_MIN, V_MAX, N_DATA)*VOLTAGE_RANGE[i,pos]
            args        = (volt,topo,PATH)
            kwargs      = {
                'net_kwargs':{'add_to_path':f'_{pos}_scaled'},
                'sim_kwargs':{'save_th':10},
            }
            tasks.append((args,kwargs))
    
    batch_launch(run_static_simulation, tasks, N_PROCS)
        
if __name__ == "__main__":
    main()
