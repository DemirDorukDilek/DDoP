import numpy as np
import time
from DDOP.CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from DDOP.CorridorGeneration.RRT import rrt
from DDOP.DDoP.Optimizer import optimize_with_split
from DDOP.Visualization3d import visualize_results,visualize_results_3d_full,visualize_interactive,visualize_simulation,plot_rrt,plot_corridors
from DDOP.CorridorGeneration.Corridor_Utils import HPolyhedron


bounds = (np.array([0.,0.,0.]), np.array([10.,10.,10.]))

obstacles = [
    HPolyhedron.from_Vrep([[3,0,0],[4,0,0],[4,4,0],[3,4,0],[3,0,4],[4,0,4],[4,4,4],[3,4,4]]), # duvar 1
    HPolyhedron.from_Vrep([[3,6,0],[4,6,0],[4,10,0],[3,10,0],[3,6,4],[4,6,4],[4,10,4],[3,10,4]]), # duvar 2
    HPolyhedron.from_Vrep([[6,2,0],[8,2,0],[8,5,0],[6,5,0],[6,2,5],[8,2,5],[8,5,5],[6,5,5]]), # kutu 1
    HPolyhedron.from_Vrep([[7,7,3],[8.5,7,3],[8.5,8.5,3],[7,8.5,3],[7,7,7],[8.5,7,7],[8.5,8.5,7],[7,8.5,7]]), # havada kutu
]

start = np.array([0.5, 9.5, 2.0])
goal = np.array([9.0, 4.0, 4.0])
# np.random.seed(42)

print("Step 1: RRT Time: ", end="")
rrt_time = time.time()
path = rrt(start, goal, obstacles, bounds, max_iter=5000, step_size=0.8)
print(time.time()-rrt_time)
if path is None:
    print(" RRT failed!")
plot_rrt(path, obstacles, start, goal, bounds, save_path='rrt.png')

print("Step 2: IRIS Corridors Time: ", end="")
iris_time = time.time()
iris_result = greedy_corridor_generation(path, obstacles, bounds)
print(time.time()-iris_time)
if iris_result is None:
    print(" Corridor generation failed!")
corridors, waypoints, radii = iris_result
print("Verification:", end="")
verify_corridors(corridors, radii)
plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path='corridors.png')

hpolys = list(map(lambda x:(x.hpoly.A,x.hpoly.b),corridors))
print("Step 3: Optimization Time:", end="")
ddop_time = time.time()
opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys,10)
print(time.time()-ddop_time)
print("Total:", time.time()-rrt_time)
visualize_results_3d_full(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results.png")
visualize_interactive(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results.html")

