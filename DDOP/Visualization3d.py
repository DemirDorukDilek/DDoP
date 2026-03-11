import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
from .Utils import H2V


def _get_trajectory_points(opt, num_points=50):
    """Optimizer'dan trajectory noktalarini cek (N-boyutlu)"""
    dim = len(opt.opt)
    t_all = []
    coords = [[] for _ in range(dim)]
    t_offset = 0

    for piece_idx in range(opt.M):
        T = opt.Ts[piece_idx]
        t_local = np.linspace(0, T, num_points)

        cs = []
        for d in range(dim):
            d_vec = opt.opt[d].dstar[2*opt.S*piece_idx : 2*opt.S*(piece_idx+1)].flatten()
            cs.append(opt.opt[d].Abs[piece_idx] @ d_vec)

        for t in t_local:
            t_all.append(t + t_offset)
            t_pow = np.array([t**i for i in range(len(cs[0]))])
            for d in range(dim):
                coords[d].append(cs[d] @ t_pow)

        t_offset += T

    return np.array(t_all), [np.array(c) for c in coords]


def _draw_hpoly_2d(ax, A, b, color, alpha=0.4, label=None):
    verts = H2V(A, b)
    if verts is not None:
        poly = MplPolygon(verts, alpha=alpha, facecolor=color,
                          edgecolor='black', linewidth=1.5, label=label)
        ax.add_patch(poly)
        return verts.mean(axis=0)
    return None


def _draw_hpoly_3d(ax, A, b, color, alpha=0.3, label=None):
    verts = H2V(A, b)
    if verts is not None and len(verts) >= 4:
        hull = ConvexHull(verts)
        faces = [verts[face] for face in hull.simplices]
        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                edgecolor='black', linewidth=0.5)
        if label:
            poly.set_label(label)
        ax.add_collection3d(poly)
        return verts.mean(axis=0)
    return None


def _draw_obstacle_2d(ax, obs):
    verts = H2V(obs.A, obs.b)
    if verts is not None:
        ax.add_patch(MplPolygon(verts, closed=True,
                                fc='gray', ec='black', alpha=0.7, lw=2))


def _draw_obstacle_3d(ax, obs):
    verts = H2V(obs.A, obs.b)
    if verts is not None and len(verts) >= 4:
        hull = ConvexHull(verts)
        faces = [verts[face] for face in hull.simplices]
        ax.add_collection3d(Poly3DCollection(
            faces, alpha=0.6, facecolor='gray', edgecolor='black', linewidth=0.5))


def _set_3d_bounds(ax, bounds):
    lower, upper = bounds
    margin = 0.5
    ax.set_xlim(lower[0]-margin, upper[0]+margin)
    ax.set_ylim(lower[1]-margin, upper[1]+margin)
    ax.set_zlim(lower[2]-margin, upper[2]+margin)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')


# ==================== PLOT FUNCTIONS ====================

def plot_rrt(path, obstacles, start, goal, bounds, save_path=None):
    lower, upper = bounds
    dim = len(start)

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        for obs in obstacles:
            _draw_obstacle_2d(ax, obs)
        ax.plot(path[:,0], path[:,1], 'b.-', lw=1.5, alpha=0.6, label=f'RRT ({len(path)} pts)')
        ax.plot(*start, 'go', ms=12, zorder=10, label='Start')
        ax.plot(*goal, 'r*', ms=15, zorder=10, label='Goal')
        ax.set_xlim(lower[0]-0.5, upper[0]+0.5)
        ax.set_ylim(lower[1]-0.5, upper[1]+0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title('RRT Path')
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for obs in obstacles:
            _draw_obstacle_3d(ax, obs)
        ax.plot(path[:,0], path[:,1], path[:,2], 'b.-', lw=1.5, alpha=0.6, label=f'RRT ({len(path)} pts)')
        ax.scatter(*start, color='green', s=100, zorder=10, label='Start')
        ax.scatter(*goal, color='red', s=150, marker='*', zorder=10, label='Goal')
        _set_3d_bounds(ax, bounds)
        ax.legend()
        ax.set_title('RRT Path')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_corridors(corridors, waypoints, radii, obstacles, start, goal, bounds, save_path=None):
    lower, upper = bounds
    dim = len(start)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(corridors), 1)))

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
        for obs in obstacles:
            _draw_obstacle_2d(ax, obs)
        for i, cor in enumerate(corridors):
            _draw_hpoly_2d(ax, cor.hpoly.A, cor.hpoly.b, colors[i], alpha=0.2, label=f'C{i}')
        for i, r in enumerate(radii):
            if r > 0:
                ax.add_patch(plt.Circle(waypoints[i+1], r, fc='red', ec='darkred',
                                        alpha=0.2, lw=1.5, zorder=4))
        ax.plot(waypoints[:,0], waypoints[:,1], 'ko-', lw=2.5, ms=10,
                label='Waypoints', zorder=5)
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
    else:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        for obs in obstacles:
            _draw_obstacle_3d(ax, obs)
        for i, cor in enumerate(corridors):
            _draw_hpoly_3d(ax, cor.hpoly.A, cor.hpoly.b, colors[i], alpha=0.15, label=f'C{i}')
        ax.plot(waypoints[:,0], waypoints[:,1], waypoints[:,2], 'ko-', lw=2.5, ms=8,
                label='Waypoints', zorder=5)
        for i, w in enumerate(waypoints):
            ax.text(w[0], w[1], w[2], f'  wp{i}', fontsize=8, fontweight='bold')
        ax.scatter(*start, color='green', s=100, zorder=10, label='Start')
        ax.scatter(*goal, color='red', s=150, marker='*', zorder=10, label='Goal')
        _set_3d_bounds(ax, bounds)
        ax.legend(fontsize=9)
        ax.set_title(f'{len(corridors)} IRIS Corridors + Chebyshev Waypoints')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_results(opt, polyhedron, waypoints_init, bounds, obstacles=None, save_path=None):
    """
    Optimize edilmis trajectory + polyhedron gorsellestirme.
    2D: XY + X(t) + Y(t)
    3D: XYZ + X(t) + Y(t) + Z(t)
    """
    dim = len(opt.opt)
    t, coords = _get_trajectory_points(opt)
    wp_opt = np.array(opt.waypoints)
    wp_init = np.array(waypoints_init)

    t_wp = [0]
    for T in opt.Ts:
        t_wp.append(t_wp[-1] + T)

    colors_poly = plt.cm.Set3(np.linspace(0, 1, len(polyhedron)))
    axis_labels = ['X', 'Y', 'Z']
    axis_colors = ['b', 'g', 'r']

    if dim == 2:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # XY trajectory
        ax1 = axes[0]
        for idx, (A, b) in enumerate(polyhedron):
            center = _draw_hpoly_2d(ax1, A, b, colors_poly[idx])
            if center is not None:
                ax1.annotate(f'P{idx}', center, fontsize=11,
                            ha='center', fontweight='bold', color='darkblue')
        if obstacles:
            for obs in obstacles:
                _draw_obstacle_2d(ax1, obs)

        ax1.plot(wp_init[:,0], wp_init[:,1], 'b--', linewidth=1.5,
                alpha=0.5, marker='s', markersize=6, label='Initial')
        ax1.plot(coords[0], coords[1], 'r-', linewidth=2.5, label='Optimized')
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
        ax1.set_title('XY Trajectory', fontweight='bold')
        ax1.autoscale()
        ax1.margins(0.1)

        # X(t), Y(t)
        for d in range(2):
            ax = axes[d+1]
            ax.plot(t, coords[d], f'{axis_colors[d]}-', linewidth=2, label=f'{axis_labels[d]}(t)')
            for i, (tw, wp) in enumerate(zip(t_wp, wp_opt)):
                ax.axvline(tw, color='gray', linestyle='--', alpha=0.5)
                ax.scatter(tw, wp[d], c='red', s=80, zorder=5, edgecolors='black')
                ax.annotate(f'q{i}', (tw, wp[d]), xytext=(5, 10),
                           textcoords='offset points', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{axis_labels[d]} (m)')
            ax.set_title(f'{axis_labels[d]}(t)', fontweight='bold')

    else:  # 3D
        fig = plt.figure(figsize=(20, 10))

        # 3D trajectory
        ax1 = fig.add_subplot(1, dim+1, 1, projection='3d')
        for idx, (A, b) in enumerate(polyhedron):
            _draw_hpoly_3d(ax1, A, b, colors_poly[idx], alpha=0.15)
        if obstacles:
            for obs in obstacles:
                _draw_obstacle_3d(ax1, obs)

        ax1.plot(wp_init[:,0], wp_init[:,1], wp_init[:,2], 'b--',
                linewidth=1.5, alpha=0.5, marker='s', markersize=5, label='Initial')
        ax1.plot(coords[0], coords[1], coords[2], 'r-', linewidth=2.5, label='Optimized')
        ax1.scatter(wp_opt[:,0], wp_opt[:,1], wp_opt[:,2], c='red', s=80,
                   zorder=5, edgecolors='black', linewidths=1.5)
        for i, wp in enumerate(wp_opt):
            ax1.text(wp[0], wp[1], wp[2], f'  q{i}', fontsize=9, fontweight='bold')

        if bounds:
            _set_3d_bounds(ax1, bounds)
        ax1.legend(loc='upper left')
        ax1.set_title('3D Trajectory', fontweight='bold')

        # X(t), Y(t), Z(t)
        for d in range(dim):
            ax = fig.add_subplot(1, dim+1, d+2)
            ax.plot(t, coords[d], f'{axis_colors[d]}-', linewidth=2, label=f'{axis_labels[d]}(t)')
            for i, (tw, wp) in enumerate(zip(t_wp, wp_opt)):
                ax.axvline(tw, color='gray', linestyle='--', alpha=0.5)
                ax.scatter(tw, wp[d], c='red', s=80, zorder=5, edgecolors='black')
                ax.annotate(f'q{i}', (tw, wp[d]), xytext=(5, 10),
                           textcoords='offset points', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{axis_labels[d]} (m)')
            ax.set_title(f'{axis_labels[d]}(t)', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# Backward compat
visualize_results_2d = visualize_results


def visualize_state(polyhedron, waypoints, obstacles=None, bounds=None):
    """Polyhedron ve waypoint gorsellestirme (optimizasyon oncesi)"""
    wp = np.array(waypoints)
    dim = wp.shape[1]
    colors = plt.cm.Set3(np.linspace(0, 1, len(polyhedron)))

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        for idx, (A, b) in enumerate(polyhedron):
            center = _draw_hpoly_2d(ax, A, b, colors[idx])
            if center is not None:
                ax.annotate(f'P{idx}', center, fontsize=12,
                            ha='center', fontweight='bold', color='darkblue')
        if obstacles:
            for obs in obstacles:
                _draw_obstacle_2d(ax, obs)
        ax.plot(wp[:,0], wp[:,1], 'bo-', linewidth=2, markersize=10, label='Waypoints')
        for i, w in enumerate(waypoints):
            ax.annotate(f'q{i}', w, xytext=(8, 8),
                        textcoords='offset points', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Polyhedron & Waypoints', fontweight='bold')
        ax.autoscale()
        ax.margins(0.1)
    else:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for idx, (A, b) in enumerate(polyhedron):
            _draw_hpoly_3d(ax, A, b, colors[idx], alpha=0.2)
        if obstacles:
            for obs in obstacles:
                _draw_obstacle_3d(ax, obs)
        ax.plot(wp[:,0], wp[:,1], wp[:,2], 'bo-', linewidth=2, markersize=8, label='Waypoints')
        for i, w in enumerate(wp):
            ax.text(w[0], w[1], w[2], f'  q{i}', fontsize=10, fontweight='bold')
        if bounds:
            _set_3d_bounds(ax, bounds)
        ax.legend(loc='upper left')
        ax.set_title('Polyhedron & Waypoints', fontweight='bold')

    plt.tight_layout()
    plt.show()
