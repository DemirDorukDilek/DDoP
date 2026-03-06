from .DDoP.Optimizer import optimize_with_split
from .CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from .CorridorGeneration.RRTstar import rrt_star
from .CorridorGeneration.Corridor_Utils import ConvexObstacle
from .Visualization import visualize_results_2d,plot_rrt,plot_corridors
from .Utils import H2V
import numpy as np
import time
import pygame

class PygameSimulator:
    """
    Pygame ile hızlı ve responsive simülasyon
    """
    
    def __init__(self, opt, polyhedra=None, obstacles=None):
        self.opt = opt
        self.polyhedra = polyhedra
        self.obstacles = obstacles
        
        # State
        self._compute_trajectory()
        self.current_time = 0.0
        self.running = True
        self.paused = False
        self.speed = 1.0
        
        # Pygame
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Trajectory Simulation')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # Coordinate transform
        self._setup_transform()
        
        # Trail
        self.trail = []
        self.max_trail = 100
    
    def _compute_trajectory(self):
        """Trajectory verilerini hesapla"""
        S2 = 2 * self.opt.S
        
        self.piece_data = []
        for piece_idx in range(self.opt.M):
            c = np.array([
                subopt.Abs[piece_idx] @ subopt.dstar[2*self.opt.S*piece_idx : 2*self.opt.S*(piece_idx+1)].flatten()
                for subopt in self.opt.opt
            ])
            dc = c[:, 1:] * np.arange(1, S2)
            
            self.piece_data.append({
                'c': c, 'dc': dc, 'T': self.opt.Ts[piece_idx]
            })
        
        self.piece_start_times = np.concatenate([[0], np.cumsum(self.opt.Ts)])
        self.total_time = self.piece_start_times[-1]
    
    def _setup_transform(self):
        """Dünya → ekran koordinat dönüşümü"""
        waypoints = np.array(self.opt.waypoints)
        
        x_min, x_max = waypoints[:, 0].min() - 10, waypoints[:, 0].max() + 10
        y_min, y_max = waypoints[:, 1].min() - 10, waypoints[:, 1].max() + 10
        
        self.scale = min(self.width / (x_max - x_min), self.height / (y_max - y_min)) * 0.8
        self.offset = np.array([
            self.width / 2 - (x_min + x_max) / 2 * self.scale,
            self.height / 2 + (y_min + y_max) / 2 * self.scale
        ])
    
    def world_to_screen(self, pos):
        """Dünya → ekran"""
        return (
            int(pos[0] * self.scale + self.offset[0]),
            int(-pos[1] * self.scale + self.offset[1])
        )
    
    def get_state_at_time(self, t):
        """t anındaki state"""
        piece_idx = 0
        for i in range(self.opt.M):
            if t >= self.piece_start_times[i] and t < self.piece_start_times[i + 1]:
                piece_idx = i
                break
        else:
            piece_idx = self.opt.M - 1
        
        t_local = np.clip(t - self.piece_start_times[piece_idx], 0, self.piece_data[piece_idx]['T'])
        
        c = self.piece_data[piece_idx]['c']
        dc = self.piece_data[piece_idx]['dc']
        S2 = c.shape[1]
        
        t_pow = np.power(t_local, np.arange(S2))
        pos = c @ t_pow
        vel = dc @ t_pow[:S2-1]
        
        return pos, vel, piece_idx
    
    def handle_events(self):
        """Event handling"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                
                elif event.key == pygame.K_r:
                    self.current_time = 0.0
                    self.trail = []
                
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False
                
                elif event.key == pygame.K_1:
                    self.speed = 1.0
                
                elif event.key == pygame.K_2:
                    self.speed = 2.0
                
                elif event.key == pygame.K_3:
                    self.speed = 0.5
                
                elif event.key == pygame.K_LEFT:
                    self.current_time = max(0, self.current_time - 0.5)
                
                elif event.key == pygame.K_RIGHT:
                    self.current_time = min(self.total_time, self.current_time + 0.5)
    
    def update(self, dt):
        """State güncelle"""
        if not self.paused:
            self.current_time += dt * self.speed
            
            if self.current_time >= self.total_time:
                self.current_time = self.total_time
        
        pos, vel, piece = self.get_state_at_time(self.current_time)
        
        # Trail
        screen_pos = self.world_to_screen(pos)
        self.trail.append(screen_pos)
        if len(self.trail) > self.max_trail:
            self.trail.pop(0)
        
        return pos, vel, piece
    
    def draw(self, pos, vel, piece):
        """Çiz"""
        self.screen.fill((30, 30, 40))
        
        # Polyhedra
        if self.polyhedra:
            for idx, (A, b) in enumerate(self.polyhedra):
                verts = H2V(A, b)
                if verts is not None:
                    screen_verts = [self.world_to_screen(v) for v in verts]
                    pygame.draw.polygon(self.screen, (70, 70, 100), screen_verts, 0)
                    pygame.draw.polygon(self.screen, (100, 100, 140), screen_verts, 2)
        
        # Obstacles
        if self.obstacles:
            for obs in self.obstacles:
                verts = np.array(obs.vertices)
                screen_verts = [self.world_to_screen(v) for v in verts]
                pygame.draw.polygon(self.screen, (200, 50, 50), screen_verts, 0)
                pygame.draw.polygon(self.screen, (150, 30, 30), screen_verts, 2)
        
        # Waypoints
        for i, w in enumerate(self.opt.waypoints):
            screen_w = self.world_to_screen(w)
            pygame.draw.circle(self.screen, (50, 200, 50), screen_w, 10)
            pygame.draw.circle(self.screen, (30, 150, 30), screen_w, 10, 2)
            
            label = self.font.render(f'q{i}', True, (255, 255, 255))
            self.screen.blit(label, (screen_w[0] + 12, screen_w[1] - 8))
        
        # Trail
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (100, 150, 255), False, self.trail, 3)
        
        # Drone
        screen_pos = self.world_to_screen(pos)
        pygame.draw.circle(self.screen, (255, 255, 0), screen_pos, 15)
        pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, 15, 3)
        
        # Velocity arrow
        vel_scale = 20
        vel_end = (
            int(screen_pos[0] + vel[0] * vel_scale),
            int(screen_pos[1] - vel[1] * vel_scale)
        )
        pygame.draw.line(self.screen, (0, 255, 0), screen_pos, vel_end, 3)
        
        # Info
        status = "PAUSED" if self.paused else "RUNNING"
        info_lines = [
            f"Status: {status}",
            f"Time: {self.current_time:.2f}s / {self.total_time:.2f}s",
            f"Speed: {self.speed}x",
            f"Piece: {piece}",
            f"Pos: ({pos[0]:.2f}, {pos[1]:.2f})",
            f"Vel: {np.linalg.norm(vel):.2f} m/s",
            "",
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset",
            "1/2/3 - Speed",
            "←/→ - Seek",
            "Q/ESC - Quit"
        ]
        
        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20))
        
        pygame.display.flip()
    
    def run(self):
        """Ana loop"""
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            
            self.handle_events()
            pos, vel, piece = self.update(dt)
            self.draw(pos, vel, piece)
        
        pygame.quit()

    @staticmethod
    def start(opt,polyhedra,obstacles):
        sim = PygameSimulator(opt, polyhedra, obstacles)
        sim.run()

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
