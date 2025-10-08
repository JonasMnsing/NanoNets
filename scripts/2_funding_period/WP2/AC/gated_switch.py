from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation, sinusoidal_voltages

# ─── Configuration ────────────────────────────────────────────────────────────────
N_NP            = 9
STAT_SIZE_BASE  = 10
N_PERIODS       = 100
SAMPLE_P_PERIOD = 40
AMPLITUDE       = 0.02
FREQUENCY       = 28.0
N_CONTROL       = 3
CONTROL_VALUES  = [-0.010,-0.008,-0.006,-0.004,-0.002,0.002,0.004,0.006,0.008,0.01]
# OUTPUT_DIR      = Path("/scratch/j_mens07/data/")
OUTPUT_DIR      = Path("/home/j/j_mens07/phd/NanoNets/scripts/2_funding_period/WP2/AC/data/gated_switch/")
LOG_LEVEL       = logging.INFO
CPU_CNT         = 10
ELECTRODE_TYPE  = ['constant']*8
TOPO            = {"Nx": N_NP,"Ny": N_NP,
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
                        "electrode_type": ELECTRODE_TYPE}
# ────────────────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=LOG_LEVEL,
                        format="%(asctime)s %(levelname)s: %(message)s")
    OUTPUT_DIR.mkdir(exist_ok=True)

    f0_hz       = FREQUENCY * 1e6
    dt          = 1 / (SAMPLE_P_PERIOD * f0_hz)
    T_sim       = N_PERIODS / f0_hz
    N_steps     = int(np.ceil(T_sim / dt))
    stat_size   = max(STAT_SIZE_BASE, int(np.round(300*FREQUENCY/5.0)))

    time_steps, volt = sinusoidal_voltages(N_steps, TOPO,
                    amplitudes=[AMPLITUDE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    frequencies=[f0_hz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], time_step=dt)
    
    tasks = []
    for Uc in CONTROL_VALUES:
                
        volt_new = volt.copy()
        volt_new[:,N_CONTROL] = Uc
                                
        # output directory per frequency
        args    = (time_steps, volt_new, TOPO, OUTPUT_DIR)
        kwargs  = {
            'net_kwargs': {'add_to_path' : f"_{FREQUENCY:.3f}_{AMPLITUDE:.3f}_{Uc:.3f}"},
            'sim_kwargs': {'stat_size':stat_size, 'save':True}
        }
        tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
