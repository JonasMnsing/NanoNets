from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40
# FREQUENCIES     = [0.06,0.12,0.25,0.5,1.,2.,5.,10.,20.,28.,44.]
FREQUENCIES     = [0.06,0.12,0.25,0.5,1.,2.,5.,6.,8.,10,12.,15.,18.,23.,28.,36.,44.,55.,68.,86.]
AMPLITUDE       = 0.02
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/AC/res_disorder/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 32
MEAN_R2         = 200.0
N_J_TOTAL       = 2*N_NP*(N_NP-1)
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    
    # Setup base directory once
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    topo    = {"Nx": N_NP,"Ny": N_NP,
               "e_pos": [[(N_NP-1)//2, 0],
                         [0, 0],
                         [N_NP-1, 0],
                         [0, (N_NP-1)//2],
                         [N_NP-1, (N_NP-1)//2],
                         [0, N_NP-1],
                         [N_NP-1, N_NP-1],
                         [(N_NP-1)//2, N_NP-1]],
                "electrode_type": ['constant']*8}
    tasks = []
    for F0 in FREQUENCIES:
        # Generate base waveform
        f0_hz       = F0 * 1e6
        dt          = 1 / (SAMPLE_P_PERIOD * f0_hz)
        T_sim       = N_PERIODS / f0_hz
        N_steps     = int(np.ceil(T_sim / dt))
        stat_size   = max(STAT_SIZE_BASE, int(np.round(300*F0/5.0)))


        time_steps, base_volt = sinusoidal_voltages(
            N_steps, topo,
            amplitudes=[AMPLITUDE,0.,0.,0.,0.,0.,0.,0.],
            frequencies=[f0_hz,0.,0.,0.,0.,0.,0.,0.],
            time_step=dt
        )
        
        for i in range(CPU_CNT):
            res_info2   = {'N':N_J_TOTAL//3,'mean_R':MEAN_R2,'std_R':0.0}
            args        = (time_steps, base_volt, topo, OUTPUT_DIR)
            kwargs      = {
                'net_kwargs': {"add_to_path" : f"_{F0:.3f}_mean2_{MEAN_R2}_{i}", "res_info2":res_info2, "seed":i},
                'sim_kwargs': {'stat_size' : stat_size, 'save':True}
            }
            tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()