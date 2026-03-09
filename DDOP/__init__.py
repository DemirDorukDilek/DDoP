from .DDoP.Optimizer import optimize_with_split
from .CorridorGeneration.IRIS import greedy_corridor_generation,verify_corridors
from .CorridorGeneration.RRTstar import rrt_star
from .CorridorGeneration.Corridor_Utils import ConvexObstacle
from .Visualization import visualize_results_2d,plot_rrt,plot_corridors
from .Utils import H2V
import numpy as np
import time
import threading
import pygame

class PygameSimulator:
    """
    Interaktif Pygame simülasyon.
    Haritaya tıklayarak hedef belirle, araç otomatik trajectory hesaplayıp uçar.
    States: IDLE -> CALCULATING -> FLYING -> IDLE
    """

    def __init__(self, start_pos, obstacles, bounds):
        self.obstacles = obstacles
        self.bounds = bounds
        self.current_pos = np.array(start_pos, dtype=float)

        # State machine
        self.state = "IDLE"
        self.goal_pos = None
        self.opt = None
        self.polyhedra = None
        self.piece_data = None
        self.piece_start_times = None
        self.current_time = 0.0
        self.total_time = 0.0

        # Threading
        self._calc_thread = None
        self._calc_result = None

        # Pygame
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Trajectory Simulation')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self._setup_transform()

        self.trail = []
        self.max_trail = 500
        self.running = True

    def _setup_transform(self):
        lower, upper = self.bounds
        margin = 1.0
        x_min, x_max = lower[0] - margin, upper[0] + margin
        y_min, y_max = lower[1] - margin, upper[1] + margin

        self.scale = min(self.width / (x_max - x_min), self.height / (y_max - y_min)) * 0.8
        self.offset = np.array([
            self.width / 2 - (x_min + x_max) / 2 * self.scale,
            self.height / 2 + (y_min + y_max) / 2 * self.scale
        ])

    def world_to_screen(self, pos):
        return (
            int(pos[0] * self.scale + self.offset[0]),
            int(-pos[1] * self.scale + self.offset[1])
        )

    def screen_to_world(self, screen_pos):
        x = (screen_pos[0] - self.offset[0]) / self.scale
        y = -(screen_pos[1] - self.offset[1]) / self.scale
        return np.array([x, y])

    def _compute_trajectory(self):
        S2 = 2 * self.opt.S
        self.piece_data = []
        for piece_idx in range(self.opt.M):
            c = np.array([
                subopt.Abs[piece_idx] @ subopt.dstar[2*self.opt.S*piece_idx : 2*self.opt.S*(piece_idx+1)].flatten()
                for subopt in self.opt.opt
            ])
            dc = c[:, 1:] * np.arange(1, S2)
            self.piece_data.append({'c': c, 'dc': dc, 'T': self.opt.Ts[piece_idx]})
        self.piece_start_times = np.concatenate([[0], np.cumsum(self.opt.Ts)])
        self.total_time = self.piece_start_times[-1]

    def get_state_at_time(self, t):
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

    def _run_calculation(self, goal):
        try:
            result = optimal_traj(self.current_pos, goal, self.obstacles, self.bounds, rrt_args = {"goal_bias":0.15})
            self._calc_result = result
        except (Exception, SystemExit):
            self._calc_result = None

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    self.running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.state == "IDLE":
                    goal = self.screen_to_world(event.pos)
                    lower, upper = self.bounds
                    if (lower[0] <= goal[0] <= upper[0] and
                        lower[1] <= goal[1] <= upper[1]):
                        self.goal_pos = goal
                        self.state = "CALCULATING"
                        self._calc_result = None
                        self._calc_thread = threading.Thread(
                            target=self._run_calculation, args=(goal,))
                        self._calc_thread.start()

    def update(self, dt):
        if self.state == "CALCULATING":
            if self._calc_thread is not None and not self._calc_thread.is_alive():
                if self._calc_result is not None:
                    opt, Ts, op_wp, opt_hpolys, obstacles, bounds = self._calc_result
                    self.opt = opt
                    self.polyhedra = opt_hpolys
                    self._compute_trajectory()
                    self.current_time = 0.0
                    self.trail = []
                    self.state = "FLYING"
                else:
                    self.state = "IDLE"
                    self.goal_pos = None
                self._calc_thread = None

        elif self.state == "FLYING":
            self.current_time += dt
            if self.current_time >= self.total_time:
                self.current_time = self.total_time
                pos, vel, piece = self.get_state_at_time(self.current_time)
                self.current_pos = pos.copy()
                self.state = "IDLE"
                return pos, vel, piece

            pos, vel, piece = self.get_state_at_time(self.current_time)
            self.current_pos = pos.copy()
            screen_pos = self.world_to_screen(pos)
            self.trail.append(screen_pos)
            if len(self.trail) > self.max_trail:
                self.trail.pop(0)
            return pos, vel, piece

        return self.current_pos, np.zeros(2), 0

    def draw(self, pos, vel, piece):
        self.screen.fill((30, 30, 40))

        # Bounds
        lower, upper = self.bounds
        corners = [
            (lower[0], lower[1]), (upper[0], lower[1]),
            (upper[0], upper[1]), (lower[0], upper[1])
        ]
        screen_corners = [self.world_to_screen(c) for c in corners]
        pygame.draw.polygon(self.screen, (80, 85, 95), screen_corners, 0)
        pygame.draw.polygon(self.screen, (120, 120, 150), screen_corners, 2)

        # Polyhedra
        if self.polyhedra:
            mouse_pos = pygame.mouse.get_pos()
            mouse_world = self.screen_to_world(mouse_pos)
            hover_idx = -1
            if self.state == "IDLE":
                for idx, (A, b) in enumerate(self.polyhedra):
                    if np.all(A @ mouse_world <= b + 1e-6):
                        hover_idx = idx
                        break

            for idx, (A, b) in enumerate(self.polyhedra):
                verts = H2V(A, b)
                if verts is not None:
                    screen_verts = [self.world_to_screen(v) for v in verts]
                    if (self.state == "FLYING" and idx == piece) or idx == hover_idx:
                        pygame.draw.polygon(self.screen, (80, 120, 180), screen_verts, 0)
                        pygame.draw.polygon(self.screen, (140, 200, 255), screen_verts, 2)
                    else:
                        pygame.draw.polygon(self.screen, (100, 100, 140), screen_verts, 1)

        # Obstacles
        if self.obstacles:
            for obs in self.obstacles:
                verts = np.array(obs.vertices)
                screen_verts = [self.world_to_screen(v) for v in verts]
                pygame.draw.polygon(self.screen, (200, 50, 50), screen_verts, 0)
                pygame.draw.polygon(self.screen, (150, 30, 30), screen_verts, 2)

        # Waypoints + full trajectory
        if self.opt is not None and self.piece_data is not None:
            # Trajectory cizgisi
            traj_points = []
            for t_sample in np.linspace(0, self.total_time, 200):
                p, _, _ = self.get_state_at_time(t_sample)
                traj_points.append(self.world_to_screen(p))
            if len(traj_points) > 1:
                pygame.draw.lines(self.screen, (60, 60, 80), False, traj_points, 2)

            # Waypoints
            for i, w in enumerate(self.opt.waypoints):
                sw = self.world_to_screen(w)
                pygame.draw.circle(self.screen, (50, 200, 50), sw, 8)
                pygame.draw.circle(self.screen, (30, 150, 30), sw, 8, 2)
                label = self.font.render(f'q{i}', True, (255, 255, 255))
                self.screen.blit(label, (sw[0] + 10, sw[1] - 8))

        # Goal marker
        if self.goal_pos is not None:
            gs = self.world_to_screen(self.goal_pos)
            pygame.draw.circle(self.screen, (255, 50, 50), gs, 12, 3)
            pygame.draw.line(self.screen, (255, 50, 50), (gs[0]-8, gs[1]), (gs[0]+8, gs[1]), 2)
            pygame.draw.line(self.screen, (255, 50, 50), (gs[0], gs[1]-8), (gs[0], gs[1]+8), 2)

        # Trail
        if len(self.trail) > 1:
            pygame.draw.lines(self.screen, (100, 150, 255), False, self.trail, 3)

        # Drone
        screen_pos = self.world_to_screen(pos)
        pygame.draw.circle(self.screen, (255, 255, 0), screen_pos, 15)
        pygame.draw.circle(self.screen, (0, 0, 0), screen_pos, 15, 3)

        # Velocity arrow
        if self.state == "FLYING" and np.linalg.norm(vel) > 0.01:
            vel_scale = 20
            vel_end = (
                int(screen_pos[0] + vel[0] * vel_scale),
                int(screen_pos[1] - vel[1] * vel_scale)
            )
            pygame.draw.line(self.screen, (0, 255, 0), screen_pos, vel_end, 3)

        # Info
        info_lines = [
            f"Status: {self.state}",
            f"Pos: ({pos[0]:.2f}, {pos[1]:.2f})",
        ]
        if self.state == "FLYING":
            info_lines.extend([
                f"Time: {self.current_time:.2f}s / {self.total_time:.2f}s",
                f"Vel: {np.linalg.norm(vel):.2f} m/s",
                f"Piece: {piece}",
            ])
        elif self.state == "CALCULATING":
            info_lines.append("Computing trajectory...")

        info_lines.extend(["", "Click - Set target", "Q/ESC - Quit"])

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20))

        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            pos, vel, piece = self.update(dt)
            self.draw(pos, vel, piece)
        pygame.quit()

    @staticmethod
    def start(start_pos, obstacles, bounds):
        sim = PygameSimulator(start_pos, obstacles, bounds)
        sim.run()


def caster(start,goal,obstacles,lower_bound,upper_bound):
    bounds = (np.array(lower_bound), np.array(upper_bound))
    obstacles = [ConvexObstacle(np.array(obstacle)) for obstacle in obstacles]
    start, goal = np.array(start), np.array(goal)
    return start,goal,obstacles,bounds

def optimal_traj(start,goal,obstacles,bounds,*_,seed=None,plot=False,rrt_args={},iris_args={}):
    if seed:
        np.random.seed(seed)


    print("Step 1: RRT* Time: ", end="")
    rrt_time = time.time()
    path = rrt_star(start, goal, obstacles, bounds,**rrt_args)
    print(time.time()-rrt_time)
    if path is None:
        print("  RRT* failed!")
        return None

    print("Step 2: IRIS Corridors Time: ", end="")
    iris_time = time.time()
    iris_result = greedy_corridor_generation(path, obstacles, bounds, **iris_args)
    print(time.time()-iris_time)
    if iris_result is None:
        print("  Corridor generation failed!")
        return None
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
    return opt, Ts, op_wp, opt_hpolys, obstacles, bounds
