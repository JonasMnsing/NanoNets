from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40
FREQUENCIES     = [0.06,0.12,0.25,0.5,1.,2.,5.,10.,20.,28.,44.]
AMPLITUDE       = 0.02
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/AC/res_disorder/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 11
MEAN_R2         = 50.0
JUNCTIONS       = [(20,21),(21,22),(22,23),(23,24),
                   (29,30),(30,31),(31,32),(32,33),
                   (38,39),(39,40),(40,41),(41,42),
                   (47,48),(48,49),(49,50),(50,51),
                   (56,57),(57,58),(58,59),(59,60),
                   (20,29),(29,38),(38,47),(47,56),
                   (21,30),(30,39),(39,48),(48,57),
                   (22,31),(31,40),(40,49),(49,58),
                   (23,32),(32,41),(41,50),(50,59),
                   (24,33),(33,42),(42,51),(51,60)]
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
        
        res_info2   = {'junctions':JUNCTIONS,'mean_R':MEAN_R2,'std_R':0.0}
        args        = (time_steps, base_volt, topo, OUTPUT_DIR)
        kwargs      = {
            'net_kwargs': {"add_to_path" : f"_{F0:.3f}_mean2_{MEAN_R2}_center", "res_info2":res_info2},
            'sim_kwargs': {'stat_size' : stat_size, 'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()