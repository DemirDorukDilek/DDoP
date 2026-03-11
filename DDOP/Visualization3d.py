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


def _project_hpoly(A, b, axes_2d):
    """3D H-polytope'u 2D eksene project et (vertex project)"""
    verts = H2V(A, b)
    if verts is None or len(verts) < 3:
        return None
    return verts[:, axes_2d]


def _draw_projected_poly(ax, A, b, axes_2d, color, alpha=0.3, label=None):
    """3D polytope'u 2D eksene project edip ciz"""
    proj = _project_hpoly(A, b, axes_2d)
    if proj is not None and len(proj) >= 3:
        try:
            hull = ConvexHull(proj)
            hull_verts = proj[hull.vertices]
            poly = MplPolygon(hull_verts, alpha=alpha, facecolor=color,
                              edgecolor='black', linewidth=1, label=label)
            ax.add_patch(poly)
        except Exception:
            pass


def _draw_projected_obstacle(ax, obs, axes_2d):
    """3D obstacle'i 2D eksene project edip ciz"""
    proj = _project_hpoly(obs.A, obs.b, axes_2d)
    if proj is not None and len(proj) >= 3:
        try:
            hull = ConvexHull(proj)
            hull_verts = proj[hull.vertices]
            ax.add_patch(MplPolygon(hull_verts, closed=True,
                                    fc='gray', ec='black', alpha=0.5, lw=1.5))
        except Exception:
            pass


def visualize_results_3d_full(opt, polyhedron, waypoints_init, bounds, obstacles=None, save_path=None):
    """
    7 panelli 3D gorsellestirme:
      Ust satir:  3D gorunum | XY kesiti | YZ kesiti | XZ kesiti
      Alt satir:  X(t)       | Y(t)      | Z(t)
    """
    t, coords = _get_trajectory_points(opt)
    wp_opt = np.array(opt.waypoints)
    wp_init = np.array(waypoints_init)

    t_wp = [0]
    for T in opt.Ts:
        t_wp.append(t_wp[-1] + T)

    colors_poly = plt.cm.Set3(np.linspace(0, 1, len(polyhedron)))
    axis_labels = ['X', 'Y', 'Z']
    axis_colors = ['b', 'g', 'r']

    fig = plt.figure(figsize=(24, 12))

    # --- 1) 3D gorunum (ust sol, buyuk) ---
    ax3d = fig.add_subplot(2, 4, 1, projection='3d')
    for idx, (A, b) in enumerate(polyhedron):
        _draw_hpoly_3d(ax3d, A, b, colors_poly[idx], alpha=0.15)
    if obstacles:
        for obs in obstacles:
            _draw_obstacle_3d(ax3d, obs)
    ax3d.plot(wp_init[:,0], wp_init[:,1], wp_init[:,2], 'b--',
              linewidth=1.5, alpha=0.5, marker='s', markersize=4, label='Initial')
    ax3d.plot(coords[0], coords[1], coords[2], 'r-', linewidth=2.5, label='Optimized')
    ax3d.scatter(wp_opt[:,0], wp_opt[:,1], wp_opt[:,2], c='red', s=60,
                 zorder=5, edgecolors='black', linewidths=1.5)
    for i, wp in enumerate(wp_opt):
        ax3d.text(wp[0], wp[1], wp[2], f'  q{i}', fontsize=8, fontweight='bold')
    if bounds:
        _set_3d_bounds(ax3d, bounds)
    ax3d.legend(loc='upper left', fontsize=8)
    ax3d.set_title('3D Trajectory', fontweight='bold')

    # --- 2-4) 2D kesitler: XY, YZ, XZ ---
    projections = [
        ([0, 1], 'XY'),
        ([1, 2], 'YZ'),
        ([0, 2], 'XZ'),
    ]
    for p_idx, (axes_2d, name) in enumerate(projections):
        ax = fig.add_subplot(2, 4, p_idx + 2)
        d0, d1 = axes_2d

        for idx, (A, b) in enumerate(polyhedron):
            _draw_projected_poly(ax, A, b, axes_2d, colors_poly[idx], alpha=0.2)
        if obstacles:
            for obs in obstacles:
                _draw_projected_obstacle(ax, obs, axes_2d)

        ax.plot(np.array(wp_init)[:,d0], np.array(wp_init)[:,d1], 'b--',
                linewidth=1.5, alpha=0.5, marker='s', markersize=4)
        ax.plot(coords[d0], coords[d1], 'r-', linewidth=2, label='Optimized')
        ax.scatter(wp_opt[:,d0], wp_opt[:,d1], c='red', s=60,
                   zorder=5, edgecolors='black', linewidths=1.5)
        for i, wp in enumerate(wp_opt):
            ax.annotate(f'q{i}', (wp[d0], wp[d1]), xytext=(5, 5),
                        textcoords='offset points', fontsize=8, fontweight='bold')

        if bounds:
            lower, upper = bounds
            margin = 0.5
            ax.set_xlim(lower[d0]-margin, upper[d0]+margin)
            ax.set_ylim(lower[d1]-margin, upper[d1]+margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f'{axis_labels[d0]} (m)')
        ax.set_ylabel(f'{axis_labels[d1]} (m)')
        ax.set_title(f'{name} Projection', fontweight='bold')

    # --- 5-7) Zaman grafikleri: X(t), Y(t), Z(t) ---
    for d in range(3):
        ax = fig.add_subplot(2, 4, d + 5)
        ax.plot(t, coords[d], f'{axis_colors[d]}-', linewidth=2, label=f'{axis_labels[d]}(t)')
        for i, (tw, wp) in enumerate(zip(t_wp, wp_opt)):
            ax.axvline(tw, color='gray', linestyle='--', alpha=0.5)
            ax.scatter(tw, wp[d], c='red', s=60, zorder=5, edgecolors='black')
            ax.annotate(f'q{i}', (tw, wp[d]), xytext=(5, 8),
                        textcoords='offset points', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(f'{axis_labels[d]} (m)')
        ax.set_title(f'{axis_labels[d]}(t)', fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_interactive(opt, polyhedron, waypoints_init, bounds, obstacles=None, save_path=None):
    """
    Plotly ile interaktif 3D gorsellestirme.
    Mouse ile dondur, zoom yap, hover ile koordinat gor.
    Legenddan polytope/obstacle/trajectory toggle et.
    save_path verilirse HTML olarak kaydeder.
    """
    import plotly.graph_objects as go

    t, coords = _get_trajectory_points(opt, num_points=100)
    wp_opt = np.array(opt.waypoints)
    wp_init = np.array(waypoints_init)

    t_wp = [0]
    for T in opt.Ts:
        t_wp.append(t_wp[-1] + T)

    fig = go.Figure()

    colors_poly = [
        f'hsla({int(i * 360 / max(len(polyhedron), 1))}, 70%, 60%, 0.25)'
        for i in range(len(polyhedron))
    ]

    # Polytope'lar
    for idx, (A, b) in enumerate(polyhedron):
        verts = H2V(A, b)
        if verts is not None and len(verts) >= 4:
            try:
                hull = ConvexHull(verts)
                x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
                i_f, j_f, k_f = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=i_f, j=j_f, k=k_f,
                    color=colors_poly[idx],
                    flatshading=True,
                    name=f'P{idx}',
                    legendgroup=f'poly{idx}',
                    showlegend=True,
                    hoverinfo='name',
                ))
            except Exception:
                pass

    # Engeller
    if obstacles:
        for oi, obs in enumerate(obstacles):
            verts = H2V(obs.A, obs.b)
            if verts is not None and len(verts) >= 4:
                try:
                    hull = ConvexHull(verts)
                    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
                    i_f, j_f, k_f = hull.simplices[:, 0], hull.simplices[:, 1], hull.simplices[:, 2]
                    fig.add_trace(go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i_f, j=j_f, k=k_f,
                        color='rgba(180, 60, 60, 0.6)',
                        flatshading=True,
                        name=f'Obstacle {oi}',
                        legendgroup='obstacles',
                        showlegend=(oi == 0),
                        legendgrouptitle_text='Obstacles' if oi == 0 else None,
                        hoverinfo='name',
                    ))
                except Exception:
                    pass

    # Initial waypoints
    fig.add_trace(go.Scatter3d(
        x=wp_init[:, 0], y=wp_init[:, 1], z=wp_init[:, 2],
        mode='lines+markers',
        line=dict(color='blue', width=3, dash='dash'),
        marker=dict(size=4, color='blue', symbol='square'),
        name='Initial',
        legendgroup='initial',
    ))

    # Optimized trajectory
    hover_text = [f't={ti:.2f}s<br>X={coords[0][i]:.2f}<br>Y={coords[1][i]:.2f}<br>Z={coords[2][i]:.2f}'
                  for i, ti in enumerate(t)]
    fig.add_trace(go.Scatter3d(
        x=coords[0], y=coords[1], z=coords[2],
        mode='lines',
        line=dict(color='red', width=5),
        name='Optimized',
        legendgroup='optimized',
        hovertext=hover_text,
        hoverinfo='text',
    ))

    # Optimized waypoints
    wp_hover = [f'q{i}<br>X={wp[0]:.2f}<br>Y={wp[1]:.2f}<br>Z={wp[2]:.2f}<br>t={t_wp[i]:.2f}s'
                for i, wp in enumerate(wp_opt)]
    fig.add_trace(go.Scatter3d(
        x=wp_opt[:, 0], y=wp_opt[:, 1], z=wp_opt[:, 2],
        mode='markers+text',
        marker=dict(size=6, color='red', line=dict(width=2, color='black')),
        text=[f'q{i}' for i in range(len(wp_opt))],
        textposition='top center',
        textfont=dict(size=10, color='black'),
        name='Waypoints',
        legendgroup='waypoints',
        hovertext=wp_hover,
        hoverinfo='text',
    ))

    # Start / Goal
    fig.add_trace(go.Scatter3d(
        x=[wp_opt[0, 0]], y=[wp_opt[0, 1]], z=[wp_opt[0, 2]],
        mode='markers', marker=dict(size=10, color='green', symbol='diamond'),
        name='Start', showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[wp_opt[-1, 0]], y=[wp_opt[-1, 1]], z=[wp_opt[-1, 2]],
        mode='markers', marker=dict(size=10, color='red', symbol='x'),
        name='Goal', showlegend=True,
    ))

    # 3D scene ayarlari
    scene = dict(
        aspectmode='data',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
    )
    if bounds:
        lower, upper = bounds
        margin = 0.5
        scene.update(
            xaxis=dict(range=[lower[0]-margin, upper[0]+margin]),
            yaxis=dict(range=[lower[1]-margin, upper[1]+margin]),
            zaxis=dict(range=[lower[2]-margin, upper[2]+margin]),
        )

    fig.update_layout(
        scene=scene,
        height=800,
        width=1200,
        title='DDoP - Interactive 3D Trajectory',
        legend=dict(x=1.02, y=1, font=dict(size=10)),
        hovermode='closest',
    )

    if save_path:
        if save_path.endswith('.html'):
            fig.write_html(save_path)
        else:
            fig.write_html(save_path.rsplit('.', 1)[0] + '.html')

    fig.show()


def visualize_simulation(opt, polyhedron, waypoints_init, bounds, obstacles=None,
                         num_frames=120, trail=True, save_path=None):
    """
    Plotly animasyonu ile drone simulasyonu.
    Play/pause butonu + zaman slider ile kontrol.
    Drone yol boyunca ilerler, arkasinda iz birakir.
    """
    import plotly.graph_objects as go

    t_all, coords = _get_trajectory_points(opt, num_points=500)
    wp_opt = np.array(opt.waypoints)

    t_wp = [0]
    for T in opt.Ts:
        t_wp.append(t_wp[-1] + T)

    # Frame indeksleri (500 noktadan num_frames kadar seciyoruz)
    total_pts = len(t_all)
    frame_indices = np.linspace(0, total_pts - 1, num_frames, dtype=int)

    # --- Statik trace'ler (her frame'de ayni) ---
    static_traces = []

    # Polytope'lar
    colors_poly = [
        f'hsla({int(i * 360 / max(len(polyhedron), 1))}, 70%, 60%, 0.2)'
        for i in range(len(polyhedron))
    ]
    for idx, (A, b) in enumerate(polyhedron):
        verts = H2V(A, b)
        if verts is not None and len(verts) >= 4:
            try:
                hull = ConvexHull(verts)
                static_traces.append(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                    color=colors_poly[idx], flatshading=True,
                    name=f'P{idx}', showlegend=True, hoverinfo='name',
                ))
            except Exception:
                pass

    # Engeller
    if obstacles:
        for oi, obs in enumerate(obstacles):
            verts = H2V(obs.A, obs.b)
            if verts is not None and len(verts) >= 4:
                try:
                    hull = ConvexHull(verts)
                    static_traces.append(go.Mesh3d(
                        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                        i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                        color='rgba(180, 60, 60, 0.6)', flatshading=True,
                        name=f'Obstacle {oi}', showlegend=(oi == 0), hoverinfo='name',
                    ))
                except Exception:
                    pass

    # Waypoints
    static_traces.append(go.Scatter3d(
        x=wp_opt[:, 0], y=wp_opt[:, 1], z=wp_opt[:, 2],
        mode='markers+text',
        marker=dict(size=4, color='gray', line=dict(width=1, color='black')),
        text=[f'q{i}' for i in range(len(wp_opt))],
        textposition='top center', textfont=dict(size=9, color='gray'),
        name='Waypoints', showlegend=True, hoverinfo='text',
    ))

    # Full trajectory (soluk)
    static_traces.append(go.Scatter3d(
        x=coords[0], y=coords[1], z=coords[2],
        mode='lines', line=dict(color='rgba(200,200,200,0.4)', width=2),
        name='Full Path', showlegend=True, hoverinfo='skip',
    ))

    n_static = len(static_traces)

    # --- Animated trace'ler (ilk frame) ---
    # Trail (iz)
    trail_trace = go.Scatter3d(
        x=[coords[0][0]], y=[coords[1][0]], z=[coords[2][0]],
        mode='lines', line=dict(color='red', width=4),
        name='Trail', showlegend=True, hoverinfo='skip',
    )

    # Drone marker
    drone_trace = go.Scatter3d(
        x=[coords[0][0]], y=[coords[1][0]], z=[coords[2][0]],
        mode='markers',
        marker=dict(size=8, color='lime', symbol='diamond',
                    line=dict(width=2, color='black')),
        name='Drone', showlegend=True,
        hovertext=f't=0.00s',
        hoverinfo='text',
    )

    # Time text (3D annotation yerine trace)
    time_trace = go.Scatter3d(
        x=[coords[0][0]], y=[coords[1][0]], z=[coords[2][0] + 0.5],
        mode='text', text=[f't=0.00s'],
        textfont=dict(size=12, color='black'),
        showlegend=False, hoverinfo='skip',
    )

    fig = go.Figure(data=static_traces + [trail_trace, drone_trace, time_trace])

    # --- Frames ---
    frames = []
    for fi, idx in enumerate(frame_indices):
        ti = t_all[idx]
        cx, cy, cz = coords[0][idx], coords[1][idx], coords[2][idx]

        if trail:
            trail_x = coords[0][:idx+1]
            trail_y = coords[1][:idx+1]
            trail_z = coords[2][:idx+1]
        else:
            trail_x, trail_y, trail_z = [cx], [cy], [cz]

        frame_data = [
            go.Scatter3d(x=trail_x, y=trail_y, z=trail_z),  # trail
            go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                hovertext=f't={ti:.2f}s<br>X={cx:.2f}<br>Y={cy:.2f}<br>Z={cz:.2f}',
            ),  # drone
            go.Scatter3d(
                x=[cx], y=[cy], z=[cz + 0.5],
                text=[f't={ti:.2f}s'],
            ),  # time label
        ]

        frames.append(go.Frame(
            data=frame_data,
            traces=list(range(n_static, n_static + 3)),
            name=f'f{fi}',
        ))

    fig.frames = frames

    # --- Slider ---
    slider_steps = []
    for fi, idx in enumerate(frame_indices):
        ti = t_all[idx]
        slider_steps.append(dict(
            args=[[f'f{fi}'], dict(frame=dict(duration=0, redraw=True), mode='immediate')],
            label=f'{ti:.1f}s',
            method='animate',
        ))

    # --- Layout ---
    scene = dict(
        aspectmode='data',
        xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)',
    )
    if bounds:
        lower, upper = bounds
        m = 0.5
        scene.update(
            xaxis=dict(range=[lower[0]-m, upper[0]+m]),
            yaxis=dict(range=[lower[1]-m, upper[1]+m]),
            zaxis=dict(range=[lower[2]-m, upper[2]+m]),
        )

    total_time_ms = t_all[-1] * 1000
    frame_duration = max(30, int(total_time_ms / num_frames))

    fig.update_layout(
        scene=scene,
        height=800, width=1100,
        title='DDoP - 3D Drone Simulation',
        legend=dict(x=1.02, y=1, font=dict(size=10)),
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            x=0.05, y=0.02,
            xanchor='left', yanchor='bottom',
            buttons=[
                dict(label='▶ Play',
                     method='animate',
                     args=[None, dict(
                         frame=dict(duration=frame_duration, redraw=True),
                         fromcurrent=True,
                         transition=dict(duration=0),
                     )]),
                dict(label='⏸ Pause',
                     method='animate',
                     args=[[None], dict(
                         frame=dict(duration=0, redraw=False),
                         mode='immediate',
                     )]),
                dict(label='⏮ Reset',
                     method='animate',
                     args=[['f0'], dict(
                         frame=dict(duration=0, redraw=True),
                         mode='immediate',
                     )]),
            ],
        )],
        sliders=[dict(
            active=0,
            steps=slider_steps,
            x=0.05, len=0.9,
            xanchor='left',
            y=0, yanchor='top',
            pad=dict(b=10, t=60),
            currentvalue=dict(prefix='Time: ', visible=True, xanchor='center'),
            transition=dict(duration=0),
        )],
    )

    if save_path:
        out = save_path if save_path.endswith('.html') else save_path.rsplit('.', 1)[0] + '.html'
        fig.write_html(out)
        print(f"Simulation saved: {out}")

    fig.show()


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
    plt.savefig("akame.png", dpi=480, bbox_inches='tight')
    plt.show()
