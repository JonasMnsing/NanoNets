import logging
import numpy as np
from pathlib import Path

# Nanonets
from nanonets.utils import batch_launch, run_static_simulation

# ─── Configuration ───
N_PROCS     = 10
LOG_LEVEL   = logging.INFO
PATH        = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/iv_curves/network_res_disorder/")
SIM_DIC     = {
    "n_trajectories" : 100,
    "sim_time"       : 1e-6,
    "eq_time"        : 2e-6,
    "ac_time"        : 40e-9,
    "max_jumps"      : 4000,
    "max_eq_jumps"   : 6000
}

# Network
N_E         = 8
L           = 9
N_J_TOTAL   = 2*L*(L-1)
N_NETS      = 32
# R_VALUES    = [50,100,200,400,800,1600]
R_VALUES    = [100,1600]
TOPOLOGY    = {"Nx": L,"Ny": L, "e_pos": [
                [(L-1)//2, 0],[0, 0],[L-1, 0],
                [0, (L-1)//2],[L-1, (L-1)//2],
                [0, L-1],[L-1, L-1],[(L-1)//2, L-1]],
                "electrode_type": ['constant']*N_E}

# Voltage
V_INPUT_MAX     = 0.1
N_INPUTS        = 300
VOLTAGE         = np.zeros(shape=(N_INPUTS,N_E+1))
VOLTAGE[:,0]    = np.round(np.linspace(-V_INPUT_MAX, V_INPUT_MAX, N_INPUTS),4)

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for MEAN_R2 in R_VALUES:
        res_info2 = {'N':N_J_TOTAL//3, 'mean_R':MEAN_R2, 'std_R':0.0}
        for i in range(N_NETS):
            args        = (VOLTAGE,TOPOLOGY,PATH)
            kwargs      = {'net_kwargs': {'add_to_path' : f"_mean2_{MEAN_R2}_{i}", "res_info2":res_info2, "seed":i},
                            "sim_kwargs":{"sim_dic":SIM_DIC,"save_th":10}}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()