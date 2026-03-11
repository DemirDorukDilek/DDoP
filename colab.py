
from DDOP import caster
from DDOP.DDoP.Optimizer import optimize_with_split
from DDOP.CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from DDOP.CorridorGeneration.RRT import rrt
from DDOP.Visualization import visualize_results,plot_rrt,plot_corridors
import time


lower,upper = [0.,0.], [10.,10.]

obstacles = [
    [[3.,0.],[4.,0.],[4.,4.],[3.,4.]],
    [[3.,6.],[4.,6.],[4.,10.],[3.,10.]],
    [[6.,2.],[8.,3.],[7.5,5.],[5.5,4.]],
    [[7.,7.],[8.5,7.],[8.5,8.5],[7.,8.5]]
]
start,goal = [0.5,9.5],[8.,2.5]

start, goal, obstacles, bounds = caster(start, goal, obstacles, lower, upper)


# np.random.seed(seed)


print("Step 1: RRT* Time: ", end="")
rrt_time = time.time()
path = rrt(start, goal, obstacles, bounds)
print(time.time()-rrt_time)
if path is None:
    print("  RRT* failed!")
plot_rrt(path, obstacles, start, goal, bounds, save_path='rrt.png')

print("Step 2: IRIS Corridors Time: ", end="")
iris_time = time.time()
iris_result = greedy_corridor_generation(path, obstacles, bounds)
print(time.time()-iris_time)
if iris_result is None:
    print("  Corridor generation failed!")
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
visualize_results_2d(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results.png")