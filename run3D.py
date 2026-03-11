from DDOP.CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from DDOP.CorridorGeneration.RRT import rrt
from DDOP.DDoP.Optimizer import optimize_with_split

from DDOP.Visualization3d import visualize_results,visualize_results_3d_full,visualize_interactive,visualize_simulation,plot_rrt,plot_corridors
from DDOP import map_reader

import time

map_path = "maps/3d_init_map.json"
start,goal,obstacles, bounds = map_reader(map_path)

print("Step 1: RRT Time: ", end="")
rrt_time = time.time()
path = rrt(start, goal, obstacles, bounds, max_iter=5000, step_size=0.8)
print(time.time()-rrt_time)
if path is None:
    print(" RRT failed!")
plot_rrt(path, obstacles, start, goal, bounds, save_path='results/rrt.png')

print("Step 2: IRIS Corridors Time: ", end="")
iris_time = time.time()
iris_result = greedy_corridor_generation(path, obstacles, bounds)
print(time.time()-iris_time)
if iris_result is None:
    print(" Corridor generation failed!")
corridors, waypoints, radii = iris_result
print("Verification:", end="")
verify_corridors(corridors, radii)
plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path='results/corridors.png')

hpolys = list(map(lambda x:(x.hpoly.A,x.hpoly.b),corridors))
print("Step 3: Optimization Time:", end="")
ddop_time = time.time()
opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys)
print(time.time()-ddop_time)
print("Total:", time.time()-rrt_time)
visualize_results_3d_full(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.png")
visualize_interactive(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.html")

