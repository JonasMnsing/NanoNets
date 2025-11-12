from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 13
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40

AMPLITUDE_LIST  = [0.02,0.03,0.04,0.05,0.06,0.07,0.08]
# FREQ_LIST_MHZ   = [0.01,0.03,0.06,6.,8.,12.,15.,18.,28.,36.,44.,55.,68.,86.,105.,133.]
FREQ_LIST_MHZ   = [0.005,0.01,0.03,0.06,0.12,0.25,0.5,1.,2.,5.,6.,8.,
                   10.,12.,15.,18.,23.,28.,36.,44.,55.,68.,86.,105.,133.,266.,610.,1200.,2400.]
OUTPUT_DIR      = Path("/scratch/j_mens07/data/2_funding_period/dynamic/AC/amp_freq/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 32
# ELECTRODE_TYPES = [['constant']*8, ['constant']*7 + ['floating']]
ELECTRODE_TYPES = [['constant']*8]
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    tasks = []
    for electrode_type in ELECTRODE_TYPES:
        topo = {"Nx": N_NP,"Ny": N_NP,
           "e_pos": [
               [(N_NP-1)//2, 0],
               [0, 0],
               [N_NP-1, 0],
               [0, (N_NP-1)//2],
               [N_NP-1, (N_NP-1)//2],
               [0, N_NP-1],
               [N_NP-1, N_NP-1],
               [(N_NP-1)//2, N_NP-1]
               ],
            "electrode_type": electrode_type}
        for amp in AMPLITUDE_LIST:
            for freq_mhz in FREQ_LIST_MHZ:
                f0_hz       = freq_mhz * 1e6
                dt          = 1 / (SAMPLE_P_PERIOD * f0_hz)
                T_sim       = N_PERIODS / f0_hz
                N_steps     = int(np.ceil(T_sim / dt))
                stat_size   = max(STAT_SIZE_BASE, int(np.round(300*freq_mhz/5.0)))

                time_steps, volt = sinusoidal_voltages(
                    N_steps, topo,
                    amplitudes=[amp, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    frequencies=[f0_hz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    time_step=dt)
                
                # output directory per frequency
                args    = (time_steps, volt, topo, OUTPUT_DIR)
                kwargs  = {
                    'net_kwargs': {'add_to_path' : f"_{freq_mhz:.3f}_{amp:.3f}_{electrode_type[-1]}"},
                    'sim_kwargs': {'stat_size':stat_size, 'save':True}
                }
                tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
