from DDOP import optimal_traj_fw, map_reader

map_path = "maps/2d_init_fw_map.json"
start, goal, obstacles, bounds = map_reader(map_path)
optimal_traj_fw(
    start, goal, obstacles, bounds,
    plot=True, interactive=True,
    rrt_args={"max_iter": 1_000_000, "step_size": 10.0},
    optimizer_args={
        "v_min": 0.0,
        "v_max": 50.0,
        "a_max": 20.0,
        "phi_max_deg": 35.0,
        "gamma_max_deg": 20.0,
        "gamma_min_deg": -30.0,
    }
)
