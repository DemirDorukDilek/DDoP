from .CorridorGeneration.Corridor_Utils import HPolyhedron
from .Utils import H2V
from . import optimal_traj
import numpy as np
import threading
import pygame
import json
import os

DEFAULT_PARAMS = {
    "seed": None,
    "rrt_args": {"max_iter": 100000},
    "iris_args": {},
    "optimizer_args": {},
}

def _flatten_params(params, prefix=""):
    items = []
    for k, v in params.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.extend(_flatten_params(v, key))
        else:
            items.append((key, v))
    return items

def _set_nested(params, dotkey, value):
    keys = dotkey.split(".")
    d = params
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

def _parse_value(text):
    text = text.strip()
    if text.lower() == "none":
        return None
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


class PygameSimulator:
    """
    Interaktif Pygame simulasyon.
    Haritaya tiklayarak hedef belirle, arac otomatik trajectory hesaplayip ucar.
    States: IDLE -> CALCULATING -> FLYING -> IDLE

    Controls:
        Click   - Set target
        P       - Toggle parameter panel
        R       - Reload params from param.json
        S       - Save params to param.json
        Q/ESC   - Quit
    Panel controls:
        UP/DOWN - Select parameter
        ENTER   - Edit selected parameter (type value, ENTER to confirm, ESC to cancel)
    """

    def __init__(self, start_pos, obstacles, bounds, param=None):
        self.obstacles = obstacles
        self.bounds = bounds
        self.current_pos = np.array(start_pos, dtype=float)
        self.param = {**DEFAULT_PARAMS, **(param or {})}
        self.param_file = "param.json"

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

        # Parameter panel
        self.show_panel = False
        self.panel_selected = 0
        self.panel_editing = False
        self.panel_edit_text = ""
        self.panel_message = ""
        self.panel_message_timer = 0.0

        # Pygame
        pygame.init()
        self.width, self.height = 1200, 800
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Trajectory Simulation')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

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

    # --- Param helpers ---

    def _get_flat_params(self):
        return _flatten_params(self.param)

    def load_params(self, path=None):
        path = path or self.param_file
        if os.path.isfile(path):
            with open(path, "r") as f:
                loaded = json.load(f)
            self.param = {**DEFAULT_PARAMS, **loaded}
            self._show_message(f"Loaded: {path}")
        else:
            self._show_message(f"Not found: {path}")

    def save_params(self, path=None):
        path = path or self.param_file
        with open(path, "w") as f:
            json.dump(self.param, f, indent=2, default=str)
        self._show_message(f"Saved: {path}")

    def _show_message(self, msg):
        self.panel_message = msg
        self.panel_message_timer = 2.0

    # --- Trajectory ---

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
            result = optimal_traj(self.current_pos, goal, self.obstacles, self.bounds, **self.param)
            self._calc_result = result
        except (Exception, SystemExit):
            self._calc_result = None

    # --- Events ---

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            elif event.type == pygame.KEYDOWN:
                if self.panel_editing:
                    self._handle_edit_key(event)
                    continue

                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    if self.show_panel:
                        self.show_panel = False
                    else:
                        self.running = False
                elif event.key == pygame.K_p:
                    self.show_panel = not self.show_panel
                elif event.key == pygame.K_r:
                    self.load_params()
                elif event.key == pygame.K_s:
                    self.save_params()
                elif self.show_panel:
                    self._handle_panel_key(event)

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if not self.show_panel and self.state == "IDLE":
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

    def _handle_panel_key(self, event):
        flat = self._get_flat_params()
        if event.key == pygame.K_UP:
            self.panel_selected = max(0, self.panel_selected - 1)
        elif event.key == pygame.K_DOWN:
            self.panel_selected = min(len(flat) - 1, self.panel_selected + 1)
        elif event.key == pygame.K_RETURN:
            if flat:
                key, val = flat[self.panel_selected]
                self.panel_editing = True
                self.panel_edit_text = "" if val is None else str(val)

    def _handle_edit_key(self, event):
        if event.key == pygame.K_RETURN:
            flat = self._get_flat_params()
            key, _ = flat[self.panel_selected]
            new_val = _parse_value(self.panel_edit_text)
            _set_nested(self.param, key, new_val)
            self.panel_editing = False
            self._show_message(f"{key} = {new_val}")
        elif event.key == pygame.K_ESCAPE:
            self.panel_editing = False
        elif event.key == pygame.K_BACKSPACE:
            self.panel_edit_text = self.panel_edit_text[:-1]
        else:
            if event.unicode and event.unicode.isprintable():
                self.panel_edit_text += event.unicode

    # --- Update ---

    def update(self, dt):
        if self.panel_message_timer > 0:
            self.panel_message_timer -= dt

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

    # --- Draw ---

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
                verts = np.array(H2V(obs.A, obs.b))
                screen_verts = [self.world_to_screen(v) for v in verts]
                pygame.draw.polygon(self.screen, (200, 50, 50), screen_verts, 0)
                pygame.draw.polygon(self.screen, (150, 30, 30), screen_verts, 2)

        # Waypoints + full trajectory
        if self.opt is not None and self.piece_data is not None:
            traj_points = []
            for t_sample in np.linspace(0, self.total_time, 200):
                p, _, _ = self.get_state_at_time(t_sample)
                traj_points.append(self.world_to_screen(p))
            if len(traj_points) > 1:
                pygame.draw.lines(self.screen, (60, 60, 80), False, traj_points, 2)

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

        # Info HUD
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

        info_lines.extend(["", "Click - Set target", "P - Params", "R - Reload params", "S - Save params", "Q/ESC - Quit"])

        for i, line in enumerate(info_lines):
            text = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (10, 10 + i * 20))

        # Parameter panel
        if self.show_panel:
            self._draw_param_panel()

        # Toast message
        if self.panel_message and self.panel_message_timer > 0:
            msg_surf = self.font.render(self.panel_message, True, (255, 255, 100))
            msg_rect = msg_surf.get_rect(centerx=self.width // 2, bottom=self.height - 20)
            bg = msg_rect.inflate(20, 10)
            pygame.draw.rect(self.screen, (40, 40, 60), bg)
            pygame.draw.rect(self.screen, (255, 255, 100), bg, 1)
            self.screen.blit(msg_surf, msg_rect)

        pygame.display.flip()

    def _draw_param_panel(self):
        flat = self._get_flat_params()
        panel_w, panel_h = 400, 30 + len(flat) * 25 + 40
        panel_x = self.width - panel_w - 20
        panel_y = 20

        # Background
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((20, 20, 35, 220))
        self.screen.blit(panel_surf, (panel_x, panel_y))
        pygame.draw.rect(self.screen, (100, 120, 200), (panel_x, panel_y, panel_w, panel_h), 2)

        # Title
        title = self.font.render("Parameters (UP/DOWN, ENTER to edit)", True, (180, 200, 255))
        self.screen.blit(title, (panel_x + 10, panel_y + 8))

        # Params
        for i, (key, val) in enumerate(flat):
            y = panel_y + 35 + i * 25
            is_selected = (i == self.panel_selected)

            if is_selected:
                pygame.draw.rect(self.screen, (60, 70, 110), (panel_x + 5, y - 2, panel_w - 10, 22))

            val_str = str(val) if val is not None else "None"
            if self.panel_editing and is_selected:
                display = f"  {key}: [{self.panel_edit_text}|]"
                color = (255, 255, 100)
            else:
                display = f"  {key}: {val_str}"
                color = (255, 255, 255) if is_selected else (180, 180, 200)

            text = self.font_small.render(display, True, color)
            self.screen.blit(text, (panel_x + 10, y))

        # Footer
        footer_y = panel_y + 35 + len(flat) * 25 + 5
        footer = self.font_small.render("R: reload  S: save  ESC: close", True, (140, 140, 170))
        self.screen.blit(footer, (panel_x + 10, footer_y))

    # --- Run ---

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.handle_events()
            pos, vel, piece = self.update(dt)
            self.draw(pos, vel, piece)
        pygame.quit()

    @staticmethod
    def start(start_pos, obstacles, bounds, param=None, param_file="param.json"):
        if param is None and os.path.isfile(param_file):
            with open(param_file, "r") as f:
                param = json.load(f)
        else:
            param = {} if param is None else param
        sim = PygameSimulator(start_pos, obstacles, bounds, param=param)
        sim.param_file = param_file
        sim.run()
