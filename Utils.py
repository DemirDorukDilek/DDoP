import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import scipy.special

def visualize_trajectory_2d(opt, Ts, checkpoints_x, checkpoints_y):
    """2D trajectory çiz"""

    # Her piece için trajectory hesapla
    t_all = []
    x_all = []
    y_all = []

    t_offset = 0
    for i in range(len(Ts)):
        T = Ts[i]
        t = np.linspace(0, T, 100)

        # x koordinatı
        d_x = opt.opt[0].dstar[2*opt.S*i : 2*opt.S*(i+1)]
        c_x = opt.opt[0].Abs[i] @ d_x
        x = np.polyval(c_x[::-1], t) # coefficient order

        # y koordinatı
        d_y = opt.opt[1].dstar[2*opt.S*i : 2*opt.S*(i+1)]
        c_y = opt.opt[1].Abs[i] @ d_y
        y = np.polyval(c_y[::-1], t)

        t_all.extend(t + t_offset)
        x_all.extend(x)
        y_all.extend(y)

        t_offset += T

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # XY plot
    axes[0].plot(x_all, y_all, 'b-', linewidth=2)
    axes[0].scatter(checkpoints_x, checkpoints_y, c='red', s=100, zorder=5)
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('2D Trajectory')
    axes[0].axis('equal')
    axes[0].grid(True)

    # X vs t
    axes[1].plot(t_all, x_all, 'b-', linewidth=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('X')
    axes[1].set_title('X(t)')
    axes[1].grid(True)

    # Y vs t
    axes[2].plot(t_all, y_all, 'g-', linewidth=2)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Y')
    axes[2].set_title('Y(t)')
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

def visualize_results_2d(opt, polyhedra, waypoints_init, obstacles=None):
    """
    3 Plot: XY Trajectory + Polyhedra, X(t), Y(t)

    opt: DDoPnD optimizer (optimize edilmiş)
    polyhedra: [(A, b), (A, b), ...] tuple listesi
    waypoints_init: başlangıç waypoint'leri
    obstacles: [{'center': (x,y), 'radius': r}, ...] (opsiyonel)
    """

    # Trajectory noktaları al
    def get_trajectory_points(opt, num_points=50):
        t_all, x_all, y_all = [], [], []
        t_offset = 0

        for piece_idx in range(opt.M):
            T = opt.Ts[piece_idx]
            t_local = np.linspace(0, T, num_points)

            d_x = opt.opt[0].dstar[2*opt.S*piece_idx : 2*opt.S*(piece_idx+1)].flatten()
            c_x = opt.opt[0].Abs[piece_idx] @ d_x

            d_y = opt.opt[1].dstar[2*opt.S*piece_idx : 2*opt.S*(piece_idx+1)].flatten()
            c_y = opt.opt[1].Abs[piece_idx] @ d_y

            for t in t_local:
                x = sum(c * t**i for i, c in enumerate(c_x))
                y = sum(c * t**i for i, c in enumerate(c_y))
                t_all.append(t + t_offset)
                x_all.append(x)
                y_all.append(y)

            t_offset += T

        return np.array(t_all), np.array(x_all), np.array(y_all)

    # Polyhedron köşeleri bul
    def polyhedron_to_vertices(A, b, bounds=(-5, 10)):
        A_bounded = np.vstack([A, [1,0], [-1,0], [0,1], [0,-1]])
        b_bounded = np.hstack([b, bounds[1], -bounds[0], bounds[1], -bounds[0]])

        n = len(A_bounded)
        vertices = []

        for i in range(n):
            for j in range(i+1, n):
                A_pair = np.array([A_bounded[i], A_bounded[j]])
                b_pair = np.array([b_bounded[i], b_bounded[j]])

                try:
                    if np.abs(np.linalg.det(A_pair)) > 1e-10:
                        vertex = np.linalg.solve(A_pair, b_pair)
                        if np.all(A_bounded @ vertex <= b_bounded + 1e-10):
                            vertices.append(vertex)
                except:
                    pass

        if len(vertices) < 3:
            return None

        vertices = np.unique(np.round(vertices, 10), axis=0)
        if len(vertices) < 3:
            return None

        center = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
        return vertices[np.argsort(angles)]

    # Data
    t, x, y = get_trajectory_points(opt)
    wp_opt = np.array(opt.waypoints)
    wp_init = np.array(waypoints_init)

    # Waypoint zamanları
    t_wp = [0]
    for T in opt.Ts:
        t_wp.append(t_wp[-1] + T)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: XY + Polyhedra ---
    ax1 = axes[0]

    colors = plt.cm.Set3(np.linspace(0, 1, len(polyhedra)))
    for idx, (A, b) in enumerate(polyhedra):
        verts = polyhedron_to_vertices(A, b)
        if verts is not None:
            from matplotlib.patches import Polygon
            poly = Polygon(verts, alpha=0.4, facecolor=colors[idx],
                          edgecolor='black', linewidth=1.5)
            ax1.add_patch(poly)
            ax1.annotate(f'P{idx}', verts.mean(axis=0), fontsize=11,
                        ha='center', fontweight='bold', color='darkblue')

    if obstacles:
        for obs in obstacles:
            from matplotlib.patches import Circle
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.6)
            ax1.add_patch(circle)

    ax1.plot(wp_init[:,0], wp_init[:,1], 'b--', linewidth=1.5,
            alpha=0.5, marker='s', markersize=6, label='Başlangıç')
    ax1.plot(x, y, 'r-', linewidth=2.5, label='Optimized')
    ax1.scatter(wp_opt[:,0], wp_opt[:,1], c='red', s=100,
               zorder=5, edgecolors='black', linewidths=2)

    for i, wp in enumerate(wp_opt):
        ax1.annotate(f'q{i}', wp, xytext=(8, 8),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory + Polyhedra', fontweight='bold')
    ax1.autoscale()
    ax1.margins(0.1)

    # --- Plot 2: X(t) ---
    ax2 = axes[1]

    ax2.plot(t, x, 'b-', linewidth=2, label='X(t)')
    for i, (tw, wp) in enumerate(zip(t_wp, wp_opt)):
        ax2.axvline(tw, color='gray', linestyle='--', alpha=0.5)
        ax2.scatter(tw, wp[0], c='red', s=80, zorder=5, edgecolors='black')
        ax2.annotate(f'q{i}', (tw, wp[0]), xytext=(5, 10),
                    textcoords='offset points', fontsize=9)

    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel('Zaman (s)')
    ax2.set_ylabel('X (m)')
    ax2.set_title('X(t)', fontweight='bold')

    # --- Plot 3: Y(t) ---
    ax3 = axes[2]

    ax3.plot(t, y, 'g-', linewidth=2, label='Y(t)')
    for i, (tw, wp) in enumerate(zip(t_wp, wp_opt)):
        ax3.axvline(tw, color='gray', linestyle='--', alpha=0.5)
        ax3.scatter(tw, wp[1], c='red', s=80, zorder=5, edgecolors='black')
        ax3.annotate(f'q{i}', (tw, wp[1]), xytext=(5, 10),
                    textcoords='offset points', fontsize=9)

    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlabel('Zaman (s)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title('Y(t)', fontweight='bold')

    plt.tight_layout()
    plt.show()



def visualize_state(polyhedra, waypoints, obstacles=None):
    """
    Polyhedra ve waypoint'leri çiz (optimizasyon öncesi)

    polyhedra: [(A, b), (A, b), ...] tuple listesi
    waypoints: [(x0,y0), (x1,y1), ...]
    obstacles: [{'center': (x,y), 'radius': r}, ...] (opsiyonel)
    """

    def polyhedron_to_vertices(A, b, bounds=(-5, 10)):
        A_bounded = np.vstack([A, [1,0], [-1,0], [0,1], [0,-1]])
        b_bounded = np.hstack([b, bounds[1], -bounds[0], bounds[1], -bounds[0]])

        n = len(A_bounded)
        vertices = []

        for i in range(n):
            for j in range(i+1, n):
                A_pair = np.array([A_bounded[i], A_bounded[j]])
                b_pair = np.array([b_bounded[i], b_bounded[j]])

                try:
                    if np.abs(np.linalg.det(A_pair)) > 1e-10:
                        vertex = np.linalg.solve(A_pair, b_pair)
                        if np.all(A_bounded @ vertex <= b_bounded + 1e-10):
                            vertices.append(vertex)
                except:
                    pass

        if len(vertices) < 3:
            return None

        vertices = np.unique(np.round(vertices, 10), axis=0)
        if len(vertices) < 3:

            return None

        center = vertices.mean(axis=0)
        angles = np.arctan2(vertices[:,1] - center[1], vertices[:,0] - center[0])
        return vertices[np.argsort(angles)]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Polyhedra
    colors = plt.cm.Set3(np.linspace(0, 1, len(polyhedra)))
    for idx, (A, b) in enumerate(polyhedra):
        verts = polyhedron_to_vertices(A, b)
        if verts is not None:
            from matplotlib.patches import Polygon
            poly = Polygon(verts, alpha=0.4, facecolor=colors[idx],
                          edgecolor='black', linewidth=2)
            ax.add_patch(poly)
            ax.annotate(f'P{idx}', verts.mean(axis=0), fontsize=12,
                        ha='center', fontweight='bold', color='darkblue')

    # Engeller
    if obstacles:
        for obs in obstacles:
            from matplotlib.patches import Circle
            circle = Circle(obs['center'], obs['radius'], color='red', alpha=0.6)
            ax.add_patch(circle)

    # Waypoints
    wp = np.array(waypoints)
    ax.plot(wp[:,0], wp[:,1], 'bo-', linewidth=2, markersize=10, label='Waypoints')

    for i, w in enumerate(waypoints):
        ax.annotate(f'q{i}', w, xytext=(8, 8),
                    textcoords='offset points', fontsize=11, fontweight='bold')

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Polyhedra ve Waypoints', fontweight='bold')
    ax.autoscale()
    ax.margins(0.1)

    plt.tight_layout()
    plt.show()


def polygon_to_polyhedron(vertices):
    vertices = np.array(vertices, dtype=float)
    n = len(vertices)
    
    A = []
    b = []
    
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        
        # Kenar vektörü
        edge = p2 - p1
        
        # Normal vektör (içe dönük) - CCW için: edge'in SAĞ tarafı
        # Sağa 90° döndürme: (x, y) → (y, -x)
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal = normal / norm
        
        # Yarı-düzlem: normal @ x ≤ normal @ p1
        A.append(normal)
        b.append(np.dot(normal, p1))
    
    return np.array(A),np.array(b)