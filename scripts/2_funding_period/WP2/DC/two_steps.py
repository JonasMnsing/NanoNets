from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
N_VOLTAGES   = 50000
TIME_STEP    = 1e-11
STAT_SIZE    = 500
N_P_VALUES   = [9]
CPU_CNT      = 32
V_WRITES     = [0.01,0.02,0.03,0.04]
T_WRITES     = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8] # [13.9, 1.6, 0.3]
T_WAITS      = [0.1,0.15,0.2,0.3,0.5,0.7,1.,1.5,2.,3.,5.,7.,10.,15.,20.,30.,50.,70.,100.,150.,200.,300.,500.,700.,1000.]
OUTPUT_DIR   = Path("/scratch/j_mens07/data/2_funding_period/dynamic/DC/two_steps/")
LOG_LEVEL    = logging.INFO
# ────────────────────────────────────────────────────────────────────────────────

def paired_pulse(V_write, t_write, t_wait, dt, N_electrodes=8, input_pos=0):
    
    # Time intervals
    t1_end      = t_write
    t_wait_end  = t_write + t_wait
    t2_end      = t_write + t_wait + t_write

    # Total steps
    n_steps     = int(np.ceil(t2_end / dt))
    t_series    = np.arange(0, n_steps)*dt

    # Voltage Time Series
    V_series        = np.zeros(n_steps)
    pulse1_indices  = t_series < t1_end
    V_series[pulse1_indices] = V_write
    pulse2_indices  = t_series >= t_wait_end
    V_series[pulse2_indices] = V_write

    # Define Input Position
    voltages                = np.zeros((n_steps,N_electrodes+1))
    voltages[:,input_pos]   = V_series

    return t_series, voltages

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    tasks = []

    # Common output base for voltage sweep
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for N_P in N_P_VALUES:
        # --- 8‐electrode (8 constant) setup ---
        topo = {"Nx": N_P,"Ny": N_P,
                    "e_pos": [[(N_P-1)//2, 0],[0, 0],[N_P-1, 0],
                            [0, (N_P-1)//2],[N_P-1, (N_P-1)//2],
                            [0, N_P-1],[N_P-1, N_P-1],[(N_P-1)//2, N_P-1]],
                    "electrode_type": ['constant']*8}
        n_elec = len(topo["e_pos"])
        for V_write in V_WRITES:
            for t_write in T_WRITES:
                for t_wait in T_WAITS:
                    time_pts, volt = paired_pulse(V_write, t_write*1e-9, t_wait*1e-9, TIME_STEP, n_elec)
                    args    = (time_pts, volt, topo, OUTPUT_DIR)
                    kwargs  = {
                        'net_kwargs': {'add_to_path' : f"_{V_write:.5f}_{t_write:.5f}_{t_wait:.5f}"},
                        'sim_kwargs': {'stat_size':STAT_SIZE,'save':True}
                    }
                    tasks.append((args, kwargs))

    batch_launch(run_dynamic_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()
