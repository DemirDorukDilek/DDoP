from DDOP import optimal_traj,map_reader
import numpy as np

map_path = "maps/kan2.json"
start,goal,obstacles, bounds = map_reader(map_path)
optimal_traj(start,goal,obstacles,bounds,plot=True,interactive=False,rrt_args={"max_iter":1_000_000})

