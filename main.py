from DDoPnDm import DDoPnD

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import scipy.special

def check_piece_feasibility(opt, piece_idx, polyhedron, v_max, a_max, num_samples=50):
    T = opt.Ts[piece_idx]
    A, b = polyhedron
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


def check_all_pieces(opt, polyhedrons, v_max, a_max, num_samples=50):
    violations = []
    for m in range(opt.M):
        poly_ok, type_ = check_piece_feasibility(opt, m, polyhedrons[m], v_max, a_max, num_samples)

        if not poly_ok:
            violations.append((m,type_))

    return violations


def split_polygon(polygon, q_prev, q_next, margin=0.1):
    q_prev = np.array(q_prev)
    q_next = np.array(q_next)

    midpoint = (q_prev + q_next) / 2
    direction = q_next - q_prev
    length = np.linalg.norm(direction)
    direction = direction / length

    plane0_point = midpoint - margin * direction
    plane1_point = midpoint + margin * direction

    poly_0 = clip_polygon_by_plane(polygon, plane0_point, direction)
    poly_1 = clip_polygon_by_plane(polygon, plane1_point, -direction)

    return poly_0, poly_1, midpoint


def clip_polygon_by_plane(polygon, plane_point, plane_normal):
    polygon = np.array(polygon)
    plane_point = np.array(plane_point)
    plane_normal = np.array(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    clipped = []
    n = len(polygon)
    for i in range(n):
        curr = polygon[i]
        next_p = polygon[(i + 1) % n]

        d_curr = np.dot(curr - plane_point, plane_normal)
        d_next = np.dot(next_p - plane_point, plane_normal)

        if d_curr >= 0:
            clipped.append(curr)

        if d_curr * d_next < 0:
            t = d_curr / (d_curr - d_next)
            intersection = curr + t * (next_p - curr)
            clipped.append(intersection)

    if len(clipped) < 3:
        return polygon

    return clipped



def optimize_with_split(Ts_init, waypoints_init, polygons_init ,max_iterations=5):

    Ts = Ts_init.copy()
    waypoints = waypoints_init.copy()
    polygons = polygons_init.copy()
    rho_v = [128.0]*len(Ts)
    rho_a = [128.0]*len(Ts)
    pakka = [1.0]*len(Ts)
    last_split = -1
    for iteration in range(max_iterations):

        opt_waypoint = list(map(np.array,waypoints))
        polyhedras = list(map(polygon_to_polyhedron,polygons))
        print(pakka)
        rho_v_arr = np.array(rho_v)
        rho_a_arr = np.array(rho_a)
        pakka_arr = np.array(pakka)

        opt = DDoPnD([1.0]*(len(waypoints)-1),opt_waypoint,polyhedras,32.0,rho_v_arr,rho_a_arr,pakka_arr,False,False,3)
        T_opt, wp_opt, cost = opt.run()
        if "opt" in locals():
            visualize_results_2d(opt, polyhedras, wp_opt, [])
        # print(T_opt, wp_opt, cost)

        waypoints = [wp_opt[i] for i in range(len(wp_opt))]
        Ts = list(T_opt)

        violations = check_all_pieces(opt, polyhedras, opt.v_max, opt.a_max)[::-1]

        if len(violations) == 0:return opt, Ts, waypoints, polygons

        if len(violations) > 1 and violations[0][0] == last_split:
            piece_idx,vaolation_type = violations[1]
        else:
            piece_idx,vaolation_type = violations[0]
        last_split = piece_idx
        penalty_list = None

        q_prev = waypoints[piece_idx]
        q_next = waypoints[piece_idx + 1]

        poly_0, poly_1,new_waypoint = split_polygon(polygons[piece_idx], q_prev, q_next, 0.5)

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

        polygons[piece_idx] = poly_1
        polygons.insert(piece_idx + 1, poly_0)


    return opt, Ts, waypoints, polygons

optimize_with_split([1.0]*(len(waypoints)-1),waypoints,polygons,100)
print()