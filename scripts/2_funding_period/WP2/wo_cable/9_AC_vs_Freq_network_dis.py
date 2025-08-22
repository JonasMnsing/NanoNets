import logging
from pathlib import Path
import numpy as np
import pandas as pd
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
T_VAL       = 5.0
V_DRIVE     = 0.02
N_VOLT      = 50000
STAT_SIZE_BASE  = 10
N_PERIODS       = 40
AMPLITUDE       = 0.1
TIME_STEP   = 1e-10
OUTPUT_DIR  = Path("/home/j/j_mens07/phd/data/2_funding_period/")
LOG_LEVEL   = logging.INFO
CPU_CNT     = 10
RAD_VALS    = [np.load("scripts/2_funding_period/WP2/wo_cable/data/radius_dis.npy")[i] for i in range(CPU_CNT)]
TOPO_VALS   = [np.load("scripts/2_funding_period/WP2/wo_cable/data/topo_dis.npy")[i] for i in range(CPU_CNT)]
DIST_VALS   = [np.load("scripts/2_funding_period/WP2/wo_cable/data/dist_dis.npy")[i] for i in range(CPU_CNT)]
E_DIST_VALS = [np.load("scripts/2_funding_period/WP2/wo_cable/data/e_dist_dis.npy")[i] for i in range(CPU_CNT)]
TOPO        = {""}
FREQ_LIST_MHZ   = [133.,105.,86.,68.,55.,44.,36.,28.,23.,18.,15.,12.,10.,8.,6.,5.,2.,1.,
                   0.5,0.25,0.12,0.06,0.03,0.01,0.008,0.006,0.004,0.002,0.001]
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")
    tasks   = []

    # single output base folder
    out_base = OUTPUT_DIR / "potential" / "wo_magic_cable" / "ac_input_disorder"
    out_base.mkdir(parents=True, exist_ok=True)

    # build tasks for each topology and frequency
    for i in range(CPU_CNT):
        for freq_mhz in FREQ_LIST_MHZ:
            f0_hz       = freq_mhz * 1e6
            dt          = 1 / (40 * f0_hz)
            T_sim       = N_PERIODS / f0_hz
            N_steps     = int(np.ceil(T_sim / dt))
            stat_size   = max(STAT_SIZE_BASE, int(np.round(300 * freq_mhz / 5.0)))
            topo        = {"electrode_type":["constant"]*7+["floating"]}
            time_steps, volt = sinusoidal_voltages(N_steps,topo,
                amplitudes=[AMPLITUDE]+7*[0.0],frequencies=[f0_hz]+7*[0.0],time_step=dt)
        
            out_base.mkdir(parents=True, exist_ok=True)
            args = (time_steps, volt, "", out_base)

            kwargs  = {
                'net_kwargs': {'add_to_path' : f"_{i}_{freq_mhz:.3f}", "net_topology" : TOPO_VALS[i],
                            "dist_matrix" : DIST_VALS[i],"electrode_dist_matrix" : E_DIST_VALS[i],
                            "radius_vals" : RAD_VALS[i],"electrode_type" : ['constant']*(E_DIST_VALS[i].shape[0]-1) + ['floating']},
                'sim_kwargs': {'T_val':T_VAL,'stat_size':stat_size,'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()