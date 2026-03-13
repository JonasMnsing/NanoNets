# first line: 1
@memory.cache
def get_all_data(freq_list, path_u, n_skip, sample_p_period, n_periods):
    # Quick connectivity check for SSHFS
    if not os.path.exists(path_u):
        raise FileNotFoundError(f"Cannot access {path_u}. Is the SSHFS mount active?")

    dts, n_steps, time_dict = {}, {}, {}
    y_t_u, y_d_u, y_t_u_e = {}, {}, {}

    slice_start = n_skip * sample_p_period

    for freq in freq_list:
        # --- 1. Timing Logic ---
        dt = 1 / (40 * freq * 1e6)
        n_s = int(np.ceil((n_periods / (freq * 1e6)) / dt))

        dts[freq] = dt
        n_steps[freq] = n_s
        time_dict[freq] = dt * np.arange(n_s)

        # --- 2. Main Data ---
        f_name = f"Nx=9_Ny=9_Ne=8_{freq:.3f}_0.0195.parquet"
        df_u = pd.read_parquet(os.path.join(path_u, f_name), columns=['Observable', 'Error'])
        
        y_t_u[freq] = df_u['Observable'].values[slice_start:].copy()
        y_t_u_e[freq] = df_u['Error'].values[slice_start:].copy()
        
        # --- 3. Mean State (The Heavy Lifter) ---
        m_name = f"mean_state_Nx=9_Ny=9_Ne=8_{freq:.3f}_0.0195.parquet"
        df_mean = pd.read_parquet(os.path.join(path_u, m_name))
        
        # Slicing columns [8:] and rows [slice_start:]
        mean_slice = df_mean.iloc[slice_start:, 8:].values
        y_d_u[freq] = get_displacement_currents(mean_slice, C_US, dt)
        
        print(f"✓ Loaded & Processed: {freq} MHz")

    return dts, n_steps, time_dict, y_t_u, y_d_u, y_t_u_e
