
from Utils import visualize_trajectory_2d,polygon_to_polyhedron,visualize_state,visualize_results_2d,plot_corridors,plot_rrt
from DDoPnDm import DDoPnD
from IRIS import greedy_corridor_generation, verify_corridors
from RRTstar import rrt_star
from Ucorridor import ConvexObstacle

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import scipy.special


def check_piece_feasibility(opt, piece_idx, hpoly, v_max, a_max, num_samples=50):
    T = opt.Ts[piece_idx]
    A, b = hpoly
    S2 = 2 * opt.S

    c = np.array([subopt.Abs[piece_idx] @ subopt.dstar[2*opt.S*piece_idx : 2*opt.S*(piece_idx+1)].flatten() for subopt in opt.opt])

    dc = c[:, 1:] * np.arange(1, S2)
    ddc = dc[:, 1:] * np.arange(1, S2-1)

    t_powers = np.power(np.linspace(0, T, num_samples)[:, None], np.arange(S2))

    pos = t_powers @ c.T
    vel_mag = np.linalg.norm(t_powers[:,:S2-1] @ dc.T, axis=1)
    acc_mag = np.linalg.norm(t_powers[:,:S2-2] @ ddc.T, axis=1)

    slack = b - (A @ pos.T).T
    min_slack_per_sample = np.min(slack, axis=1)

    if np.any(slack < 0):
        return False,0

    if np.max(vel_mag) > v_max:
        return False,1

    if np.max(acc_mag) > a_max:
        return False,2

    return True,-1


def check_all_pieces(opt, hploys, v_max, a_max, num_samples=50):
    violations = []
    for m in range(opt.M):
        poly_ok, type_ = check_piece_feasibility(opt, m, hploys[m], v_max, a_max, num_samples)

        if not poly_ok:
            violations.append((m,type_))

    return violations


def clip_hpoly(A, b, plane_point, plane_normal):
    plane_normal = np.array(plane_normal, dtype=float)
    plane_point = np.array(plane_point, dtype=float)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    a_new = -plane_normal.reshape(1, -1)
    b_new = np.array([-plane_normal @ plane_point])

    A_clipped = np.vstack([A, a_new])
    b_clipped = np.concatenate([b, b_new])

    return A_clipped, b_clipped


def split_hpoly(hpoly, q_prev, q_next, margin=0.1):
    A,b = hpoly

    q_prev = np.array(q_prev, dtype=float)
    q_next = np.array(q_next, dtype=float)

    midpoint = (q_prev + q_next) / 2
    direction = q_next - q_prev
    length = np.linalg.norm(direction)
    direction = direction / length

    plane0_point = midpoint - margin * direction
    plane1_point = midpoint + margin * direction

    A0, b0 = clip_hpoly(A, b, plane0_point, direction)
    A1, b1 = clip_hpoly(A, b, plane1_point, -direction)

    return (A0, b0), (A1, b1), midpoint




def optimize_with_split(Ts_init, waypoints_init, hpolys_init ,max_iterations=5):

    Ts = Ts_init.copy()
    waypoints = waypoints_init.copy()
    hpolys = hpolys_init.copy()
    rho_v = [128.0]*len(Ts)
    rho_a = [128.0]*len(Ts)
    pakka = [1.0]*len(Ts)
    last_split = -1
    for iteration in range(max_iterations):

        opt_waypoint = list(map(np.array,waypoints))
        rho_v_arr = np.array(rho_v)
        rho_a_arr = np.array(rho_a)
        pakka_arr = np.array(pakka)

        opt = DDoPnD([1.0]*(len(waypoints)-1),opt_waypoint,hpolys,32.0,rho_v_arr,rho_a_arr,pakka_arr,False,False,3)
        T_opt, wp_opt, cost = opt.run()
        if "opt" in locals():
            pass
            # visualize_results_2d(opt, hpolys, wp_opt, [])
        # print(T_opt, wp_opt, cost)

        waypoints = [wp_opt[i] for i in range(len(wp_opt))]
        Ts = list(T_opt)

        violations = check_all_pieces(opt, hpolys, opt.v_max, opt.a_max)[::-1]
        if len(violations) == 0:return opt, Ts, waypoints, hpolys

        if len(violations) > 1 and violations[0][0] == last_split:
            piece_idx,vaolation_type = violations[1]
        else:
            piece_idx,vaolation_type = violations[0]
        last_split = piece_idx
        penalty_list = None

        print(piece_idx,vaolation_type)

        q_prev = waypoints[piece_idx]
        q_next = waypoints[piece_idx + 1]

        poly_0, poly_1,new_waypoint = split_hpoly(hpolys[piece_idx], q_prev, q_next, 0.5)

        waypoints.insert(piece_idx + 1, new_waypoint)
        Ts[piece_idx] = Ts[piece_idx] / 2
        Ts.insert(piece_idx + 1, Ts[piece_idx] / 2)

        if vaolation_type == 0:
            pakka[piece_idx] = pakka[piece_idx]*1.5
            pakka.insert(piece_idx + 1, pakka[piece_idx])
            rho_v.insert(piece_idx + 1,128.0)
            rho_a.insert(piece_idx + 1,128.0)
        elif vaolation_type == 1:
            rho_v[piece_idx] = rho_v[piece_idx]*1.5
            rho_v.insert(piece_idx + 1, rho_v[piece_idx])
            pakka.insert(piece_idx + 1,1.0)
            rho_a.insert(piece_idx + 1,128.0)
        elif vaolation_type == 2:
            rho_a[piece_idx] = rho_a[piece_idx]*1.5
            rho_a.insert(piece_idx + 1, rho_a[piece_idx])
            pakka.insert(piece_idx + 1,1.0)
            rho_v.insert(piece_idx + 1,128.0)

        hpolys[piece_idx] = poly_1
        hpolys.insert(piece_idx + 1, poly_0)


    return opt, Ts, waypoints, hpolys



import time
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
print("Verification:", end="")
verify_corridors(corridors, radii)

hpolys = list(map(lambda x:(x.hpoly.A,x.hpoly.b),corridors))
print("Step 3: Optimization Time:", end="")
ddop_time = time.time()
opt, Ts, op_wp, opt_hpolys = optimize_with_split([1.0]*(len(waypoints)-1),waypoints,hpolys,10)
print(time.time()-ddop_time)
print("Total:", time.time()-rrt_time)
plot_rrt(path, obstacles, start, goal, bounds, save_path='rrt.png')
plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path='corridors.png')
visualize_results_2d(opt, opt_hpolys, op_wp, bounds, obstacles, save_path="results.png")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.animation import FuncAnimation
import threading
import queue
import time


class TrajectorySimulator:
    """
    Eş zamanlı trajectory simülasyonu
    - Dışarıdan komut alır
    - Real-time görselleştirme
    - Trajectory re-planning
    """
    
    def __init__(self, opt, polyhedra=None, obstacles=None):
        self.opt = opt
        self.polyhedra = polyhedra
        self.obstacles = obstacles
        
        # State
        self.current_pos = np.array(opt.waypoints[0])
        self.current_vel = np.zeros(2)
        self.current_time = 0.0
        self.current_piece = 0
        
        # Simulation
        self.dt = 0.02 # 50 Hz
        self.running = False
        self.paused = False
        
        # Command queue
        self.command_queue = queue.Queue()
        
        # Trajectory data
        self._compute_trajectory()
        
        # History (iz için)
        self.position_history = [self.current_pos.copy()]
        self.max_history = 100
    
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
            ddc = dc[:, 1:] * np.arange(1, S2-1)
            
            self.piece_data.append({
                'c': c,
                'dc': dc,
                'ddc': ddc,
                'T': self.opt.Ts[piece_idx]
            })
        
        # Piece başlangıç zamanları
        self.piece_start_times = np.concatenate([[0], np.cumsum(self.opt.Ts)])
        self.total_time = self.piece_start_times[-1]
    
    def get_state_at_time(self, t):
        """t anındaki pos, vel, acc"""
        # Hangi piece?
        piece_idx = 0
        for i in range(self.opt.M):
            if t >= self.piece_start_times[i] and t < self.piece_start_times[i + 1]:
                piece_idx = i
                break
        else:
            piece_idx = self.opt.M - 1
        
        # Local time
        t_local = t - self.piece_start_times[piece_idx]
        t_local = np.clip(t_local, 0, self.piece_data[piece_idx]['T'])
        
        # Katsayılar
        c = self.piece_data[piece_idx]['c']
        dc = self.piece_data[piece_idx]['dc']
        ddc = self.piece_data[piece_idx]['ddc']
        
        S2 = c.shape[1]
        t_pow = np.power(t_local, np.arange(S2))
        
        pos = c @ t_pow
        vel = dc @ t_pow[:S2-1]
        acc = ddc @ t_pow[:S2-2]
        
        return pos, vel, acc, piece_idx
    
    def step(self):
        """Bir simülasyon adımı"""
        if self.paused:
            return
        
        self.current_time += self.dt
        
        if self.current_time >= self.total_time:
            self.current_time = self.total_time
            self.running = False
        
        pos, vel, acc, piece = self.get_state_at_time(self.current_time)
        
        self.current_pos = pos
        self.current_vel = vel
        self.current_piece = piece
        
        # History güncelle
        self.position_history.append(pos.copy())
        if len(self.position_history) > self.max_history:
            self.position_history.pop(0)
    
    def process_commands(self):
        """Komut kuyruğunu işle"""
        while not self.command_queue.empty():
            try:
                cmd = self.command_queue.get_nowait()
                self._execute_command(cmd)
            except queue.Empty:
                break
    
    def _execute_command(self, cmd):
        """Komutu çalıştır"""
        cmd_type = cmd.get('type')
        
        if cmd_type == 'pause':
            self.paused = True
            print("⏸ Paused")
        
        elif cmd_type == 'resume':
            self.paused = False
            print("▶ Resumed")
        
        elif cmd_type == 'stop':
            self.running = False
            print("⏹ Stopped")
        
        elif cmd_type == 'reset':
            self.current_time = 0.0
            self.current_pos = np.array(self.opt.waypoints[0])
            self.current_vel = np.zeros(2)
            self.current_piece = 0
            self.position_history = [self.current_pos.copy()]
            print("🔄 Reset")
        
        elif cmd_type == 'goto':
            # Yeni hedefe git (re-planning gerekir)
            target = cmd.get('target')
            print(f"🎯 New target: {target}")
            # TODO: Re-planning implementasyonu
        
        elif cmd_type == 'speed':
            # Simülasyon hızı
            factor = cmd.get('factor', 1.0)
            self.dt = 0.02 * factor
            print(f"⏩ Speed: {factor}x")
        
        elif cmd_type == 'set_time':
            # Belirli zamana atla
            t = cmd.get('time', 0.0)
            self.current_time = np.clip(t, 0, self.total_time)
            pos, vel, _, piece = self.get_state_at_time(self.current_time)
            self.current_pos = pos
            self.current_vel = vel
            self.current_piece = piece
            print(f"⏱ Time set to: {t:.2f}s")
    
    def send_command(self, cmd):
        """Komut gönder"""
        self.command_queue.put(cmd)


class SimulationGUI:
    """
    Matplotlib tabanlı simülasyon GUI
    """
    
    def __init__(self, simulator):
        self.sim = simulator
        self.fig = None
        self.ax = None
        self.anim = None
        
        # Plot objeleri
        self.drone_marker = None
        self.trail_line = None
        self.vel_arrow = None
        self.info_text = None
    
    def setup_plot(self):
        """Plot'u hazırla"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Polyhedra
        if self.sim.polyhedra:
            colors = plt.cm.Pastel1(np.linspace(0, 1, len(self.sim.polyhedra)))
            for idx, (A, b) in enumerate(self.sim.polyhedra):
                verts = polyhedron_to_vertices(A, b)
                if verts is not None:
                    polygon = MplPolygon(verts, alpha=0.25, facecolor=colors[idx],
                                        edgecolor='gray', linewidth=1.5, linestyle='--')
                    self.ax.add_patch(polygon)
        
        # Obstacles
        if self.sim.obstacles:
            for obs in self.sim.obstacles:
                verts = np.array(obs.vertices)
                if len(verts) > 0:
                    if not np.allclose(verts[0], verts[-1]):
                        verts = np.vstack([verts, verts[0]])
                    polygon = MplPolygon(verts[:-1], alpha=0.8, facecolor='red',
                                        edgecolor='darkred', linewidth=2)
                    self.ax.add_patch(polygon)
        
        # Waypoints
        waypoints = np.array(self.sim.opt.waypoints)
        self.ax.plot(waypoints[:, 0], waypoints[:, 1], 'o',
                    markersize=12, markerfacecolor='lightgreen',
                    markeredgecolor='darkgreen', markeredgewidth=2)
        
        for i, w in enumerate(waypoints):
            self.ax.annotate(f'q{i}', w, xytext=(8, 8), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        # Full trajectory (soluk)
        t_arr = np.linspace(0, self.sim.total_time, 200)
        full_path = np.array([self.sim.get_state_at_time(t)[0] for t in t_arr])
        self.ax.plot(full_path[:, 0], full_path[:, 1], '-',
                    color='lightblue', linewidth=1, alpha=0.5)
        
        # Animated objects
        self.trail_line, = self.ax.plot([], [], '-', color='blue', linewidth=2.5, alpha=0.8)
        self.drone_marker, = self.ax.plot([], [], 'o', markersize=18,
                                          markerfacecolor='yellow', markeredgecolor='black',
                                          markeredgewidth=2, zorder=10)
        self.vel_arrow = self.ax.annotate('', xy=(0, 0), xytext=(0, 0),
                                          arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      fontsize=11, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                      family='monospace')
        
        # Axis
        margin = 5
        all_pos = np.array([self.sim.get_state_at_time(t)[0] for t in np.linspace(0, self.sim.total_time, 50)])
        self.ax.set_xlim(all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin)
        self.ax.set_ylim(all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_title('Trajectory Simulation (Press H for help)', fontweight='bold')
        
        # Keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _on_key(self, event):
        """Klavye eventi"""
        key = event.key.lower()
        
        if key == ' ': # Space - pause/resume
            if self.sim.paused:
                self.sim.send_command({'type': 'resume'})
            else:
                self.sim.send_command({'type': 'pause'})
        
        elif key == 'r': # Reset
            self.sim.send_command({'type': 'reset'})
        
        elif key == 'q': # Quit
            self.sim.send_command({'type': 'stop'})
            plt.close(self.fig)
        
        elif key == '1': # Normal speed
            self.sim.send_command({'type': 'speed', 'factor': 1.0})
        
        elif key == '2': # 2x speed
            self.sim.send_command({'type': 'speed', 'factor': 2.0})
        
        elif key == '3': # 0.5x speed
            self.sim.send_command({'type': 'speed', 'factor': 0.5})
        
        elif key == 'left': # Geri sar
            new_t = max(0, self.sim.current_time - 0.5)
            self.sim.send_command({'type': 'set_time', 'time': new_t})
        
        elif key == 'right': # İleri sar
            new_t = min(self.sim.total_time, self.sim.current_time + 0.5)
            self.sim.send_command({'type': 'set_time', 'time': new_t})
        
        elif key == 'h': # Help
            print("""
╔═══════════════════════════════════════╗
║ KEYBOARD CONTROLS ║
╠═══════════════════════════════════════╣
║ SPACE │ Pause / Resume ║
║ R │ Reset ║
║ Q │ Quit ║
║ 1 │ Normal speed (1x) ║
║ 2 │ Fast speed (2x) ║
║ 3 │ Slow speed (0.5x) ║
║ ← │ Rewind 0.5s ║
║ → │ Forward 0.5s ║
║ H │ Show this help ║
╚═══════════════════════════════════════╝
            """)
    
    def update(self, frame):
        """Animasyon update"""
        self.sim.process_commands()
        self.sim.step()
        
        # Trail
        history = np.array(self.sim.position_history)
        self.trail_line.set_data(history[:, 0], history[:, 1])
        
        # Drone
        self.drone_marker.set_data([self.sim.current_pos[0]], [self.sim.current_pos[1]])
        
        # Velocity arrow
        vel_scale = 0.5
        vel_end = self.sim.current_pos + self.sim.current_vel * vel_scale
        self.vel_arrow.set_position(self.sim.current_pos)
        self.vel_arrow.xy = vel_end
        
        # Info text
        status = "⏸ PAUSED" if self.sim.paused else "▶ RUNNING"
        if not self.sim.running:
            status = "⏹ FINISHED"
        
        vel_mag = np.linalg.norm(self.sim.current_vel)
        
        self.info_text.set_text(
            f'{status}\n'
            f'Time: {self.sim.current_time:.2f}s / {self.sim.total_time:.2f}s\n'
            f'Piece: {self.sim.current_piece}\n'
            f'Pos: ({self.sim.current_pos[0]:.2f}, {self.sim.current_pos[1]:.2f})\n'
            f'Vel: {vel_mag:.2f} m/s'
        )
        
        return self.trail_line, self.drone_marker, self.info_text
    
    def run(self, interval=20):
        """Simülasyonu başlat"""
        self.setup_plot()
        self.sim.running = True
        
        self.anim = FuncAnimation(self.fig, self.update,
                                  interval=interval, blit=True,
                                  cache_frame_data=False)
        
        plt.show()


def polyhedron_to_vertices(A, b):
    """H-rep → vertices"""
    n = len(A)
    vertices = []
    
    for i in range(n):
        for j in range(i + 1, n):
            det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
            if abs(det) < 1e-10:
                continue
            
            x = (b[i] * A[j, 1] - b[j] * A[i, 1]) / det
            y = (A[i, 0] * b[j] - A[j, 0] * b[i]) / det
            vertex = np.array([x, y])
            
            if np.all(A @ vertex <= b + 1e-6):
                vertices.append(vertex)
    
    if len(vertices) < 3:
        return None
    
    vertices = np.unique(np.round(vertices, 6), axis=0)
    if len(vertices) < 3:
        return None
    
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]


# ========================================
# KULLANIM
# ========================================

def run_simulation(opt, polyhedra=None, obstacles=None):
    """Simülasyonu başlat"""
    sim = TrajectorySimulator(opt, polyhedra, obstacles)
    gui = SimulationGUI(sim)
    gui.run()
    return sim
# Network Kontrollü Versiyon
import socket
import json
import threading


class NetworkSimulator(TrajectorySimulator):
    """
    Network üzerinden komut alan simülasyon
    """
    
    def __init__(self, opt, polyhedra=None, obstacles=None, port=5555):
        super().__init__(opt, polyhedra, obstacles)
        self.port = port
        self.server_thread = None
    
    def start_server(self):
        """UDP server başlat"""
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"🌐 Server listening on UDP port {self.port}")
    
    def _server_loop(self):
        """Server loop"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('localhost', self.port))
        sock.settimeout(0.1)
        
        while self.running or not self.command_queue.empty():
            try:
                data, addr = sock.recvfrom(1024)
                cmd = json.loads(data.decode())
                self.send_command(cmd)
                print(f"📩 Received: {cmd}")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"Error: {e}")
        
        sock.close()


def send_network_command(cmd, port=5555):
    """Simülasyona komut gönder"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = json.dumps(cmd).encode()
    sock.sendto(data, ('localhost', port))
    sock.close()


# ========================================
# NETWORK KULLANIM
# ========================================

# Terminal 1: Simülasyon
# sim = NetworkSimulator(opt, polyhedra, obstacles, port=5555)
# sim.start_server()
# gui = SimulationGUI(sim)
# gui.run()

# Terminal 2: Komutlar
# send_network_command({'type': 'pause'})
# send_network_command({'type': 'resume'})
# send_network_command({'type': 'speed', 'factor': 2.0})
# send_network_command({'type': 'goto', 'target': [30, 10]})
# Pygame Versiyonu (Daha Responsive)
import pygame
import numpy as np


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
                verts = polyhedron_to_vertices(A, b)
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


# ========================================
# KULLANIM
# ========================================

def run_pygame_simulation(opt, polyhedra=None, obstacles=None):
    """Pygame simülasyonu başlat"""
    sim = PygameSimulator(opt, polyhedra, obstacles)
    sim.run()

# Matplotlib versiyonu
# sim = run_simulation(opt, opt_hpolys, obstacles)

# Pygame versiyonu (daha responsive)
run_pygame_simulation(opt, opt_hpolys, obstacles)

# Network kontrollü
# sim = NetworkSimulator(opt, opt_hpolys, obstacles, port=5555)
# sim.start_server()
# gui = SimulationGUI(sim)
# gui.run()

# # Başka terminal/process'ten komut gönder
# send_network_command({'type': 'pause'})
# send_network_command({'type': 'speed', 'factor': 2.0})
