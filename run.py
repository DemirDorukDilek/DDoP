import numpy as np

from DDOP import caster,PygameSimulator


lower,upper = [0.,0.], [10.,10.]

obstacles = [
    [[3.,0.],[4.,0.],[4.,4.],[3.,4.]],
    [[3.,6.],[4.,6.],[4.,10.],[3.,10.]],
    [[6.,2.],[8.,3.],[7.5,5.],[5.5,4.]],
    [[7.,7.],[8.5,7.],[8.5,8.5],[7.,8.5]]
]
start = [1.,5.]

start, _, obstacles, bounds = caster(start, start, obstacles, lower, upper)

PygameSimulator.start(start, obstacles, bounds)
