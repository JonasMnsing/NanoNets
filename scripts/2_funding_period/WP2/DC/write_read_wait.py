from pathlib import Path
import logging, numpy as np
from nanonets.utils import batch_launch, run_dynamic_simulation

# ─── Configuration ────────────────────────────────────────────────────────────────
TIME_STEP   = 1e-10
STAT_SIZE   = 500
V_read      = 0.005
V_write     = [20.,40.,60.,80.]
dt_write    = [1,1e1,1e2,1e3,1e4,1e5] # in ns
dt_wait     = []
N_write     = [round(dt*1e-9 / TIME_STEP) for dt in dt_write]
N_write     = 
# ────────────────────────────────────────────────────────────────────────────────
