import time
import numpy as np
import matplotlib as plt
import scipy
from scipy.optimize import minimize
import scipy.special


np.random.seed(42)
bounds = (np.array([0.,0.]), np.array([10.,10.]))

obstacles = [
    ConvexObstacle(np.array([[3.,0.],[4.,0.],[4.,4.],[3.,4.]])),
    ConvexObstacle(np.array([[3.,6.],[4.,6.],[4.,10.],[3.,10.]])),
    ConvexObstacle(np.array([[6.,2.],[8.,3.],[7.5,5.],[5.5,4.]])),
    ConvexObstacle(np.array([[7.,7.],[8.5,7.],[8.5,8.5],[7.,8.5]])),
]

start, goal = np.array([1.,5.]), np.array([9.,9.])

print("Step 1: RRT* Time: ", end="")
rrt_time = time.time()
path = rrt_star(start, goal, obstacles, bounds,max_iter=3000, step_size=0.5, goal_bias=0.15, goal_tol=0.5, safety_margin=0.1)
print(time.time()-rrt_time)
if path is None:
    print("  RRT* failed!")
    exit(3)

print("Step 2: IRIS Corridors Time: ", end="")
iris_time = time.time()
iris_result = greedy_corridor_generation(path, obstacles, bounds)
print(time.time()-iris_time)
if iris_result is None:
    print("  Corridor generation failed!")
    exit(2)
corridors, waypoints, radii = iris_result
print("Verification:")
verify_corridors(corridors, radii)