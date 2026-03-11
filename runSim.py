from DDOP import map_reader
from DDOP.Sim2D import PygameSimulator
from DDOP.Sim3D import DashSimulator

map_path = "maps/3d_init_map.json"
start, goal, obstacles, bounds = map_reader(map_path)
dim = start.shape[0]
if dim == 2:
    PygameSimulator.start(start, obstacles, bounds) # require desktop enviroment
elif dim == 3:
    DashSimulator.start(start, obstacles, bounds) # http://localhost:8050