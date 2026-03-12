from .DDoP.Optimizer import optimize_with_split
from .CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from .CorridorGeneration.RRT import rrt
from .CorridorGeneration.Corridor_Utils import HPolyhedron
from .Visualization3d import visualize_results,visualize_interactive,plot_rrt,plot_corridors,visualize_results_3d
from .Utils import H2V
import numpy as np
import time
import threading
import pygame
import os
import json

def caster(start,goal,obstacles,lower_bound,upper_bound):
    bounds = (np.array(lower_bound), np.array(upper_bound))
    obstacles = [HPolyhedron.from_Vrep(np.array(obstacle)) for obstacle in obstacles]
    start, goal = np.array(start), np.array(goal)
    return start,goal,obstacles,bounds

def map_reader(path):
    with open(path,"r",encoding="utf-8") as f:
        obj=json.load(f)
    start,goal,lower,upper,obstacles = obj["start"],obj["goal"],obj["lower"],obj["upper"],obj["obstacles"]
    return caster(start,goal,obstacles,lower,upper)


def optimal_traj(start,goal,obstacles,bounds,*_,seed=None,plot=False,rrt_args=None,iris_args={},optimizer_args={}, interactive=False):
    if seed:
        np.random.seed(seed)
    dim = start.shape[0]
    if dim == 2:
        rrt_args = {} if rrt_args is None else rrt_args
    elif dim == 3:
        rrt_args = {"max_iter":5000, "step_size":0.8} if rrt_args is None else rrt_args
    else:
        rrt_args = {} if rrt_args is None else rrt_args
        plot = False

    print("Step 1: RRT Time: ", end="")
    rrt_time = time.time()
    path = rrt(start, goal, obstacles, bounds, **rrt_args)
    print(time.time()-rrt_time)
    if path is None:
        print(" RRT failed!")
        return False
    if plot:
        plot_rrt(path, obstacles, start, goal, bounds, save_path='results/rrt.png')

    print("Step 2: IRIS Corridors Time: ", end="")
    iris_time = time.time()
    iris_result = greedy_corridor_generation(path, obstacles, bounds, **iris_args)
    print(time.time()-iris_time)
    if iris_result is None:
        print(" Corridor generation failed!")
        return False
    corridors, waypoints, radii = iris_result
    print("Verification:", end="")
    verify_corridors(corridors, radii)
    if plot:
        plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path='results/corridors.png')

    hpolys = list(map(lambda x:(x.hpoly.A,x.hpoly.b),corridors))
    print("Step 3: Optimization Time:", end="")
    ddop_time = time.time()
    opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys, **optimizer_args)
    print(time.time()-ddop_time)
    print("Total:", time.time()-rrt_time)
    if plot:
        if dim == 3:
            visualize_results_3d(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.png")
            if interactive: visualize_interactive(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.html")
        elif dim == 2: visualize_results(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results/results.png")

    return opt, Ts, op_wp, opt_hpolys, obstacles, bounds
