import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation, distribute_array_across_processes

# ─── Configuration ───
N_PROCS     = 10
LOG_LEVEL   = logging.INFO
PATH        = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/stats_eq_jumps_res_disorder/")

# Network
N_E         = 8
M_VALS      = [1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
L           = 9
R_VALS      = [100,1600]
N_J_TOTAL   = 2*L*(L-1)
N_NETS      = 32

TOPOLOGY    = {"Nx": L,"Ny": L, "e_pos": [
            [(L-1)//2, 0],[0, 0],[L-1, 0],
            [0, (L-1)//2],[L-1, (L-1)//2],
            [0, L-1],[L-1, L-1],[(L-1)//2, L-1]],
            "electrode_type": ['constant']*N_E}

# Voltage
V_INPUT_MAX     = 0.2
N_INPUTS        = 5
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(0.03, V_INPUT_MAX, N_INPUTS),4)

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for R in R_VALS:
        res_info2 = {'N':N_J_TOTAL//3, 'mean_R':R, 'std_R':0.0}
        for i in range(N_NETS):
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
                    kwargs      = {'net_kwargs': {"pack_optimizer":False, "add_to_path":f"_{M}_{R}_{i}",
                                                  "res_info2":res_info2, "seed":i},
                                    "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":1}}
                    tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()