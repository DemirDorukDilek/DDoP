import numpy as np
import pygame


from DDOP import optimal_traj,PygameSimulator


lower,upper = [0.,0.], [10.,10.]

obstacles = [
    [[3.,0.],[4.,0.],[4.,4.],[3.,4.]],
    [[3.,6.],[4.,6.],[4.,10.],[3.,10.]],
    [[6.,2.],[8.,3.],[7.5,5.],[5.5,4.]],
    [[7.,7.],[8.5,7.],[8.5,8.5],[7.,8.5]]
]

start, goal = [1.,5.], [9.,9.]
opt, Ts, op_wp, opt_hpolys, obstacles = optimal_traj(start,goal,obstacles,lower,upper,seed=42)




PygameSimulator.start(opt,opt_hpolys,obstacles)