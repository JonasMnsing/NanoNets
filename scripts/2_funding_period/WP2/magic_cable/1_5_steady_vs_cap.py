from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_static_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
U_0         = 0.1
T_VAL       = 5.0
N_VOLT      = 1001
N_P         = 9
# FOLDER      = Path("/home/j/j_mens07/phd/data/2_funding_period/")
FOLDER      = Path("/mnt/c/Users/jonas/Desktop/phd/data/2_funding_period/")
CAP_VALS    = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6]
CPU_CNT     = 10
LOG_LEVEL   = logging.INFO

# ────────────────────────────────────────────────────────────────────────────────

def main():
    """Launch DC input vs capacitance sweeps for 2- and 8-electrode setups."""
    # Setup logging
    logging.basicConfig(level=LOG_LEVEL,format="%(asctime)s %(levelname)s: %(message)s")

    # Base output path
    out_base = FOLDER / "potential" / "magic_cable" / "steady_vs_cap"
    out_base.mkdir(parents=True, exist_ok=True)

    tasks = []

    # Two-electrode (constant input + floating output)
    topo2 = {
        "Nx": N_P,
        "Ny": N_P,
        "e_pos": [[(N_P-1)//2,0], [(N_P-1)//2, N_P-1]],
        "electrode_type": ['constant', 'constant']
    }

    volt2       = np.zeros((N_VOLT, len(topo2['e_pos'])+1), float)
    volt2[:, 0] = np.linspace(-U_0, U_0, N_VOLT)

    for cap in CAP_VALS:
        # Prepare args and kwargs
        args = (volt2, topo2, out_base)
        kwargs = {'net_kwargs': {'high_C_output': True,
                                 'np_info2': {'np_index': [81],
                                              'mean_radius': cap,
                                              'std_radius': 0.0},
                                 'pack_optimizer':False,
                                 'add_to_path': f"_{cap}"},
                'sim_kwargs': {'T_val':T_VAL, 'save_th':10}}
        tasks.append((args, kwargs))

    # Eight-electrode (constant at 7, floating last)
    topo8 = {
        "Nx": N_P,
        "Ny": N_P,
        "e_pos": [[(N_P-1)//2,0],[0,0],[N_P-1,0],
                   [0,(N_P-1)//2],[N_P-1,(N_P-1)//2],
                   [0,N_P-1],[N_P-1,N_P-1],[(N_P-1)//2,N_P-1]],
        "electrode_type": ['constant']*7 + ['constant']
    }
    volt8       = np.zeros((N_VOLT, len(topo8['e_pos'])+1), float)
    volt8[:, 0] = np.linspace(-U_0, U_0, N_VOLT)

    for cap in CAP_VALS:
        args    = (volt8, topo8, out_base)
        kwargs  = {'net_kwargs': {'high_C_output': True,
                                 'np_info2': {'np_index': [81],
                                              'mean_radius': cap,
                                              'std_radius': 0.0},
                                 'pack_optimizer':False,
                                 'add_to_path': f"_{cap}"},
                    'sim_kwargs': {'T_val':T_VAL, 'save_th':10}}
        tasks.append((args, kwargs))

    # Launch all simulations in parallel
    batch_launch(run_static_simulation, tasks, CPU_CNT)

if __name__ == "__main__":
    main()