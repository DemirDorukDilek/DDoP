from .DDoP.Optimizer import optimize_with_split
from .CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from .CorridorGeneration.RRTstar import rrt_star
from .CorridorGeneration.Corridor_Utils import ConvexObstacle
from .Visualization import visualize_results_2d,plot_rrt,plot_corridors
import numpy as np
import time


def optimal_traj(start,goal,obstacles,lower_bound,upper_bound,*_,seed=None,plot=False,rrt_args={},iris_args={}):
    if seed:
        np.random.seed(seed)
    bounds = (np.array(lower_bound), np.array(upper_bound))
    obstacles = [ConvexObstacle(np.array(obstacle)) for obstacle in obstacles]
    start, goal = np.array(start), np.array(goal)

    print("Step 1: RRT* Time: ", end="")
    rrt_time = time.time()
    path = rrt_star(start, goal, obstacles, bounds,**rrt_args)
    print(time.time()-rrt_time)
    if path is None:
        print("  RRT* failed!")
        exit(3)

    print("Step 2: IRIS Corridors Time: ", end="")
    iris_time = time.time()
    iris_result = greedy_corridor_generation(path, obstacles, bounds, **iris_args)
    print(time.time()-iris_time)
    if iris_result is None:
        print("  Corridor generation failed!")
        exit(2)
    corridors, waypoints, radii = iris_result
    print("Verification:", end="")
    verify_corridors(corridors, radii)

    hpolys = list(map(lambda x:(x.hpoly.A,x.hpoly.b),corridors))
    print("Step 3: Optimization Time:", end="")
    ddop_time = time.time()
    opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys,10)
    print(time.time()-ddop_time)
    print("Total:", time.time()-rrt_time)
    if plot:
        plot_rrt(path, obstacles, start, goal, bounds, save_path='rrt.png')
        plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path='corridors.png')
        visualize_results_2d(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results.png")
    return opt, Ts, op_wp, opt_hpolys, obstacles
