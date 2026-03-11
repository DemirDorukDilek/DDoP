import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import HalfspaceIntersection, ConvexHull
from .Utils import H2V

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

def visualize_results_2d(opt, polyhedron, waypoints_init, bounds, obstacles=None, save_path=None):
    """
    3 Plot: XY Trajectory + Polyhedron, X(t), Y(t)

    opt: DDoPnD optimizer (optimize edilmiş)
    polyhedron: [(A, b), (A, b), ...] tuple listesi
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

    # --- Plot 1: XY + Polyhedron ---
    ax1 = axes[0]

    colors = plt.cm.Set3(np.linspace(0, 1, len(polyhedron)))
    for idx, (A, b) in enumerate(polyhedron):
        verts = H2V(A, b)
        if verts is not None:
            from matplotlib.patches import Polygon
            poly = Polygon(verts, alpha=0.4, facecolor=colors[idx],
                          edgecolor='black', linewidth=1.5)
            ax1.add_patch(poly)
            ax1.annotate(f'P{idx}', verts.mean(axis=0), fontsize=11,
                        ha='center', fontweight='bold', color='darkblue')

    if obstacles:
        for obs in obstacles:
            ax1.add_patch(MplPolygon(obs.vertices, closed=True,
                                 fc='gray', ec='black', alpha=0.7, lw=2))

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
    ax1.set_title('XY Trajectory + Polyhedron', fontweight='bold')
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
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def visualize_state(polyhedron, waypoints, obstacles=None):
    """
    Polyhedron ve waypoint'leri çiz (optimizasyon öncesi)

    polyhedron: [(A, b), (A, b), ...] tuple listesi
    waypoints: [(x0,y0), (x1,y1), ...]
    obstacles: [{'center': (x,y), 'radius': r}, ...] (opsiyonel)
    """

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Polyhedron
    colors = plt.cm.Set3(np.linspace(0, 1, len(polyhedron)))
    for idx, (A, b) in enumerate(polyhedron):
        verts = H2V(A, b)
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
    ax.set_title('Polyhedron ve Waypoints', fontweight='bold')
    ax.autoscale()
    ax.margins(0.1)

    plt.tight_layout()
    plt.show()


def plot_rrt(path, obstacles, start, goal, bounds, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    lower, upper = bounds

    for obs in obstacles:
        ax.add_patch(MplPolygon(H2V(obs.A,obs.b), closed=True,
                                 fc='gray', ec='black', alpha=0.7, lw=2))

    ax.plot(path[:,0], path[:,1], 'b.-', lw=1.5, alpha=0.6, label=f'RRT* ({len(path)} pts)')
    ax.plot(*start, 'go', ms=12, zorder=10, label='Start')
    ax.plot(*goal, 'r*', ms=15, zorder=10, label='Goal')

    ax.set_xlim(lower[0]-0.5, upper[0]+0.5)
    ax.set_ylim(lower[1]-0.5, upper[1]+0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('RRT* Path')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path=None):
    def hpoly_to_vertices(A, b, interior_point):
        try:
            hs = HalfspaceIntersection(np.column_stack([A, -b]), interior_point)
            hull = ConvexHull(hs.intersections)
            return hs.intersections[hull.vertices]
        except Exception:
            return None
    fig, ax = plt.subplots(figsize=(10, 10))
    lower, upper = bounds
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(corridors), 1)))

    for obs in obstacles:
        ax.add_patch(MplPolygon(H2V(obs.A,obs.b), closed=True,
                                 fc='gray', ec='black', alpha=0.7, lw=2))

    for i, cor in enumerate(corridors):
        verts = hpoly_to_vertices(cor.hpoly.A, cor.hpoly.b, cor.ellipsoid.d)
        if verts is not None:
            ax.add_patch(MplPolygon(verts, closed=True, fc=colors[i],
                                     ec=colors[i], alpha=0.2, lw=2, label=f'C{i}'))

    for i, r in enumerate(radii):
        if r > 0:
            ax.add_patch(plt.Circle(waypoints[i+1], r, fc='red', ec='darkred',
                                     alpha=0.2, lw=1.5, zorder=4))

    ax.plot(waypoints[:,0], waypoints[:,1], 'ko-', lw=2.5, ms=10,
             label='DDoPnD waypoints', zorder=5)
    for i, w in enumerate(waypoints):
        ax.annotate(f'wp{i}', w, textcoords="offset points",
                     xytext=(8, 8), fontsize=9, fontweight='bold')

    ax.plot(*start, 'go', ms=12, zorder=10)
    ax.plot(*goal, 'r*', ms=15, zorder=10)

    ax.set_xlim(lower[0]-0.5, upper[0]+0.5)
    ax.set_ylim(lower[1]-0.5, upper[1]+0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title(f'{len(corridors)} IRIS Corridors + Chebyshev Waypoints')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


