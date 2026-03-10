import numpy as np
import sys
from DDOP import map_reader,PygameSimulator

if sys.argv[1]:
    
lower,upper = [0.,0.], [10.,10.]

obstacles = [
    [[3.,0.],[4.,0.],[4.,4.],[3.,4.]],
    [[3.,6.],[4.,6.],[4.,10.],[3.,10.]],
    [[6.,2.],[8.,3.],[7.5,5.],[5.5,4.]],
    [[7.,7.],[8.5,7.],[8.5,8.5],[7.,8.5]]
]
start = [1.,5.]

start, _, obstacles, bounds = map_reader(sys.argv[1])

PygameSimulator.start(start, obstacles, bounds)
