import numpy as np
import time
from DDOP.CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from DDOP.CorridorGeneration.RRT import rrt
from DDOP.DDoP.Optimizer import optimize_with_split
from DDOP.Visualization3d import visualize_results,visualize_results_3d_full,visualize_interactive,visualize_simulation,plot_rrt,plot_corridors
from DDOP.CorridorGeneration.Corridor_Utils import HPolyhedron


bounds = (np.array([0.,0.,0.]), np.array([10.,10.,10.]))

obstacles = [
    # --- Duvar 1: x=2..3.5, yer-tavan, y=0..4 (alt) + y=6..10 (ust) => y=4..6 bosluk, ama pencere: z=2..4 ---
    HPolyhedron.from_Vrep([[2,0,0],[3.5,0,0],[3.5,4,0],[2,4,0],[2,0,10],[3.5,0,10],[3.5,4,10],[2,4,10]]),       # duvar1-alt
    HPolyhedron.from_Vrep([[2,6,0],[3.5,6,0],[3.5,10,0],[2,10,0],[2,6,10],[3.5,6,10],[3.5,10,10],[2,10,10]]),    # duvar1-ust
    HPolyhedron.from_Vrep([[2,4,0],[3.5,4,0],[3.5,6,0],[2,6,0],[2,4,2],[3.5,4,2],[3.5,6,2],[2,6,2]]),           # duvar1-pencere-alt (boslugu daralt: sadece z=2..4 acik)
    HPolyhedron.from_Vrep([[2,4,4],[3.5,4,4],[3.5,6,4],[2,6,4],[2,4,10],[3.5,4,10],[3.5,6,10],[2,6,10]]),       # duvar1-pencere-ust

    # --- Duvar 2: x=5..6.5, yer-tavan, y=0..3.5 + y=3.5..10 ama pencere y=3.5..5.5 z=5..7.5 ---
    HPolyhedron.from_Vrep([[5,0,0],[6.5,0,0],[6.5,3.5,0],[5,3.5,0],[5,0,10],[6.5,0,10],[6.5,3.5,10],[5,3.5,10]]),    # duvar2-sol
    HPolyhedron.from_Vrep([[5,5.5,0],[6.5,5.5,0],[6.5,10,0],[5,10,0],[5,5.5,10],[6.5,5.5,10],[6.5,10,10],[5,10,10]]), # duvar2-sag
    HPolyhedron.from_Vrep([[5,3.5,0],[6.5,3.5,0],[6.5,5.5,0],[5,5.5,0],[5,3.5,5],[6.5,3.5,5],[6.5,5.5,5],[5,5.5,5]]),     # duvar2-pencere-alt (acik: z=5..7.5)
    HPolyhedron.from_Vrep([[5,3.5,7.5],[6.5,3.5,7.5],[6.5,5.5,7.5],[5,5.5,7.5],[5,3.5,10],[6.5,3.5,10],[6.5,5.5,10],[5,10,10]]), # duvar2-pencere-ust

    # --- Duvar 3: x=8..9.5, yer-tavan, y=0..7 + y=8.5..10 => dar gecit y=7..8.5, z=6..8.5 ---
    HPolyhedron.from_Vrep([[8,0,0],[9.5,0,0],[9.5,7,0],[8,7,0],[8,0,10],[9.5,0,10],[9.5,7,10],[8,7,10]]),       # duvar3-alt
    HPolyhedron.from_Vrep([[8,8.5,0],[9.5,8.5,0],[9.5,10,0],[8,10,0],[8,8.5,10],[9.5,8.5,10],[9.5,10,10],[8,10,10]]), # duvar3-ust
    HPolyhedron.from_Vrep([[8,7,0],[9.5,7,0],[9.5,8.5,0],[8,8.5,0],[8,7,6],[9.5,7,6],[9.5,8.5,6],[8,8.5,6]]),   # duvar3-pencere-alt (acik: z=6..8.5)
    HPolyhedron.from_Vrep([[8,7,8.5],[9.5,7,8.5],[9.5,8.5,8.5],[8,8.5,8.5],[8,7,10],[9.5,7,10],[9.5,8.5,10],[8,8.5,10]]), # duvar3-pencere-ust
]

start = np.array([0.5, 5.0, 3.0])
goal = np.array([9.8, 7.5, 7.0])
# np.random.seed(42)

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
opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys,10)
print(time.time()-ddop_time)
print("Total:", time.time()-rrt_time)
visualize_results_3d_full(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.png")
visualize_interactive(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.html")

