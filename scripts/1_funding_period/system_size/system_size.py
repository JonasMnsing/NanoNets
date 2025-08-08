import logging
from pathlib import Path

# Nanonets
from nanonets.utils import logic_gate_sample, distribute_array_across_processes, batch_launch, run_static_simulation

# ─── Configuration ───
N_MIN, N_MAX    = 3, 16
N_E             = 8
V_CONTROL       = 0.1
V_INPUT         = 0.01
V_GATE          = 0.0
N_DATA          = 20000
N_PROCS         = 10
T_VAL           = 5
LOG_LEVEL       = logging.INFO
PATH            = Path("/mnt/c/Users/jonas/Desktop/phd/data/1_funding_period/system_size/unscaled/")
INPUT_POS       = [1,3]
# --------------------- 

def main():
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s: %(message)s")
    PATH.mkdir(parents=True, exist_ok=True)
    tasks = []
    for n in range(N_MIN, N_MAX + 1):
        topo = {"Nx": n, "Ny": n,
                "e_pos" : [[0,0], 
                           [int((n-1)/2),0],
                           [n-1,0], 
                           [0,int((n-1)/2)],
                           [0,n-1],
                           [n-1,int((n)/2)],
                           [int((n)/2),(n-1)],
                           [n-1,n-1]],
                "electrode_type" : ['constant']*N_E}
        volt = logic_gate_sample(V_CONTROL, INPUT_POS, N_DATA, topo, V_INPUT,
                                 V_GATE, sample_technique='uniform')
        for p in range(N_PROCS):
            volt_p  = distribute_array_across_processes(p, volt.copy(), N_PROCS)
            args    = (volt_p,topo,PATH)
            kwargs  = {}
            tasks.append((args,kwargs))

    batch_launch(run_static_simulation, tasks, N_PROCS)

if __name__ == "__main__":
    main()