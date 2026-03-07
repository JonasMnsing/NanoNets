import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation, distribute_array_across_processes

# ─── Configuration ───
N_PROCS     = 10
LOG_LEVEL   = logging.INFO
PATH        = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/stats_eq_jumps/")

# Network
N_E     = 8
M_VALS  = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
L_VALS  = [3,9,15]
T_VALS  = [0.1,293]

# Voltage
V_INPUT_MAX     = 0.2
N_INPUTS        = 5
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(0.03, V_INPUT_MAX, N_INPUTS),4)

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for L in L_VALS:
        TOPOLOGY = {"Nx": L,"Ny": L, "e_pos": [
            [(L-1)//2, 0],[0, 0],[L-1, 0],
            [0, (L-1)//2],[L-1, (L-1)//2],
            [0, L-1],[L-1, L-1],[(L-1)//2, L-1]],
            "electrode_type": ['constant']*N_E}
        for T in T_VALS:
            for M in M_VALS:
                SIM_DIC     = {
                    "n_trajectories" : 1000,
                    "sim_time"       : 1*1e-9,
                    "eq_time"        : 2*1e-9,
                    "ac_time"        : 40e-9,
                    "max_jumps"      : 1,
                    "max_eq_jumps"   : M,
                }
                
                for p in range(N_PROCS):
                    volt        = VOLTAGE.copy()
                    volt_p      = distribute_array_across_processes(p, volt.copy(), N_PROCS)
                    args        = (volt_p,TOPOLOGY,PATH)
                    kwargs      = {'net_kwargs': {"pack_optimizer":False, "add_to_path":f"_{M}_{T}"},
                                    "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":1,"T_val":T}}
                    tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()