from DDOP import optimal_traj,map_reader

map_path = "maps/3d_init_map.json"
start,goal,obstacles, bounds = map_reader(map_path)
optimal_traj(start,goal,obstacles,bounds,plot=True,interactive=False)

