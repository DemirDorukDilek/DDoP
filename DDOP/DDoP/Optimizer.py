from .DDoPnDm import DDoPnD
from .DDoPnDm_FW import DDoPnDFixedWing
import numpy as np
from ..Visualization3d import visualize_results,visualize_interactive,plot_rrt,plot_corridors,visualize_results_3d


def check_piece(opt, piece_idx, hpoly, v_max, a_max, num_samples=50):
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
        poly_ok, type_ = check_piece(opt, m, hploys[m], v_max, a_max, num_samples)

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


def split_hpoly(hpoly, q_prev, q_next, pmargin=0.1, split_point=None):
    A,b = hpoly

    q_prev = np.array(q_prev, dtype=float)
    q_next = np.array(q_next, dtype=float)

    if split_point is None:
        split_point = (q_prev + q_next) / 2

    direction = q_next - q_prev
    length = np.linalg.norm(direction)
    direction = direction / length
    margin = min(pmargin, 0.25 * length)

    plane0_point = split_point - margin * direction
    plane1_point = split_point + margin * direction

    A0, b0 = clip_hpoly(A, b, plane0_point, direction)
    A1, b1 = clip_hpoly(A, b, plane1_point, -direction)

    return (A0, b0), (A1, b1), split_point


def optimize_with_split(Ts_init, waypoints_init, hpolys_init, max_iterations=10,rho_t=32.0,rho_v=128.0, rho_a=128.0, pakka=1.0, **opt_args):

    Ts = Ts_init.copy()
    waypoints = waypoints_init.copy()
    hpolys = hpolys_init.copy()
    rho_v_list = [rho_v]*len(Ts)
    rho_a_list = [rho_a]*len(Ts)
    pakka_list = [pakka]*len(Ts)
    last_split = -1
    for _ in range(max_iterations):

        opt_waypoint = list(map(np.array,waypoints))
        rho_v_arr = np.array(rho_v_list)
        rho_a_arr = np.array(rho_a_list)
        pakka_arr = np.array(pakka_list)

        opt = DDoPnD(Ts,opt_waypoint,hpolys,rho_t,rho_v_arr,rho_a_arr,pakka_arr,**opt_args)
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

        print(piece_idx,vaolation_type)

        q_prev = waypoints[piece_idx]
        q_next = waypoints[piece_idx + 1]

        poly_0, poly_1,new_waypoint = split_hpoly(hpolys[piece_idx], q_prev, q_next, 0.5)

        waypoints.insert(piece_idx + 1, new_waypoint)
        Ts.insert(piece_idx + 1, Ts[piece_idx] / 2)
        Ts[piece_idx] = Ts[piece_idx] / 2

        if vaolation_type == 0:
            pakka_list.insert(piece_idx + 1, pakka_list[piece_idx])
            pakka_list[piece_idx] = pakka_list[piece_idx]*1.5
            rho_v_list.insert(piece_idx + 1,128.0)
            rho_a_list.insert(piece_idx + 1,128.0)
        elif vaolation_type == 1:
            rho_v_list.insert(piece_idx + 1, rho_v_list[piece_idx])
            rho_v_list[piece_idx] = rho_v_list[piece_idx]*1.5
            pakka_list.insert(piece_idx + 1,1.0)
            rho_a_list.insert(piece_idx + 1,128.0)
        elif vaolation_type == 2:
            rho_a_list.insert(piece_idx + 1, rho_a_list[piece_idx])
            rho_a_list[piece_idx] = rho_a_list[piece_idx]*1.5
            pakka_list.insert(piece_idx + 1,1.0)
            rho_v_list.insert(piece_idx + 1,128.0)

        hpolys[piece_idx] = poly_1
        hpolys.insert(piece_idx + 1, poly_0)


    return opt, Ts, waypoints, hpolys


def check_piece_fw(opt, piece_idx, hpoly, v_max, a_max, v_min,
                   a_lat_max, tan2_gamma_max=None, tan2_gamma_min=None,
                   num_samples=50, tol=1e-3):
    T = opt.Ts[piece_idx]
    A, b = hpoly
    S2 = 2 * opt.S
    dim = len(opt.opt)

    c = np.array([subopt.Abs[piece_idx] @ subopt.dstar[2*opt.S*piece_idx : 2*opt.S*(piece_idx+1)].flatten() for subopt in opt.opt])

    dc = c[:, 1:] * np.arange(1, S2)
    ddc = dc[:, 1:] * np.arange(1, S2-1)

    t_powers = np.power(np.linspace(0, T, num_samples)[:, None], np.arange(S2))

    pos = t_powers @ c.T
    vel = t_powers[:, :S2-1] @ dc.T          # (num_samples, dim)
    acc = t_powers[:, :S2-2] @ ddc.T          # (num_samples, dim)
    vel_mag = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)

    slack = b - (A @ pos.T).T

    if np.any(slack < 0):
        worst = np.unravel_index(np.argmin(slack), slack.shape)[0]
        return False, 0, pos[worst]

    if np.max(vel_mag) > v_max * (1 + tol):
        worst = np.argmax(vel_mag)
        return False, 1, pos[worst]

    if np.max(acc_mag) > a_max * (1 + tol):
        worst = np.argmax(acc_mag)
        return False, 2, pos[worst]

    # Fixed-wing: minimum speed
    if np.min(vel_mag) < v_min * (1 - tol):
        worst = np.argmin(vel_mag)
        return False, 3, pos[worst]

    # Fixed-wing: lateral acceleration (turn radius)
    vel_mag_safe = np.maximum(vel_mag, 1e-10)
    a_dot_v = np.sum(acc * vel, axis=1)
    a_lat_sq = acc_mag ** 2 - a_dot_v ** 2 / (vel_mag_safe ** 2)
    a_lat_sq = np.maximum(a_lat_sq, 0.0)
    if np.max(np.sqrt(a_lat_sq)) > a_lat_max * (1 + tol):
        worst = np.argmax(a_lat_sq)
        return False, 4, pos[worst]

    # Fixed-wing: flight path angle (3D only)
    if dim == 3 and (tan2_gamma_max is not None or tan2_gamma_min is not None):
        vz = vel[:, 2]
        vh_sq = vel[:, 0] ** 2 + vel[:, 1] ** 2
        climb_violation = vz ** 2 - vh_sq * (np.where(vz > 0, tan2_gamma_max or 1e10, tan2_gamma_min or 1e10))
        if np.max(climb_violation) > 0:
            worst = np.argmax(climb_violation)
            return False, 5, pos[worst]

    return True, -1, None


def check_all_pieces_fw(opt, hpolys, v_max, a_max, v_min, a_lat_max,
                        tan2_gamma_max=None, tan2_gamma_min=None,
                        num_samples=50):
    violations = []
    for m in range(opt.M):
        ok, type_, viol_pos = check_piece_fw(opt, m, hpolys[m], v_max, a_max,
                                              v_min, a_lat_max,
                                              tan2_gamma_max, tan2_gamma_min,
                                              num_samples)
        if not ok:
            violations.append((m, type_, viol_pos))
    return violations

def optimize_with_split_fw(Ts_init, waypoints_init, hpolys_init,
                           max_iterations=100, rho_t=32.0,
                           rho_v=128.0, rho_a=128.0, pakka=1.0,
                           v_min=10.0, phi_max_deg=35.0,
                           gamma_max_deg=20.0, gamma_min_deg=-30.0,
                           rho_vmin=128.0, rho_turn=128.0, rho_climb=128.0,
                           v_init=None, v_final=None,
                           **opt_args):
    Ts = Ts_init.copy()
    waypoints = waypoints_init.copy()
    hpolys = hpolys_init.copy()
    rho_v_list = [rho_v] * len(Ts)
    rho_a_list = [rho_a] * len(Ts)
    pakka_list = [pakka] * len(Ts)
    rho_vmin_list = [rho_vmin] * len(Ts)
    rho_turn_list = [rho_turn] * len(Ts)
    rho_climb_list = [rho_climb] * len(Ts)

    g_acc = 9.81
    a_lat_max = g_acc * np.tan(np.radians(phi_max_deg))
    tan2_gamma_max = np.tan(np.radians(gamma_max_deg)) ** 2
    tan2_gamma_min = np.tan(np.radians(abs(gamma_min_deg))) ** 2

    last_split = -1
    opt = None
    for _ in range(max_iterations):

        opt_waypoint = list(map(np.array, waypoints))
        rho_v_arr = np.array(rho_v_list)
        rho_a_arr = np.array(rho_a_list)
        pakka_arr = np.array(pakka_list)
        rho_vmin_arr = np.array(rho_vmin_list)
        rho_turn_arr = np.array(rho_turn_list)
        rho_climb_arr = np.array(rho_climb_list)

        opt = DDoPnDFixedWing(
            Ts, opt_waypoint, hpolys,
            rho_t, rho_v_arr, rho_a_arr, pakka_arr,
            v_min=v_min, phi_max_deg=phi_max_deg,
            gamma_max_deg=gamma_max_deg, gamma_min_deg=gamma_min_deg,
            rho_vmin=rho_vmin_arr, rho_turn=rho_turn_arr, rho_climb=rho_climb_arr,
            v_init=v_init, v_final=v_final,
            **opt_args
        )
        T_opt, wp_opt, cost = opt.run()

        visualize_results(opt, hpolys, wp_opt, [], [], save_path=f'results/{_}.png')

        waypoints = [wp_opt[i] for i in range(len(wp_opt))]
        Ts = list(T_opt)

        violations = check_all_pieces_fw(
            opt, hpolys, opt.v_max, opt.a_max,
            v_min, a_lat_max, tan2_gamma_max, tan2_gamma_min
        )[::-1]

        if len(violations) == 0:
            return opt, Ts, waypoints, hpolys

        # Separate split-worthy violations from penalty-only violations
        # v_min (type 3): penalty only (split makes short segments worse)
        # turn (type 4): split (adds waypoint so optimizer sees mid-segment constraint)
        split_violations = [(m, t, p) for m, t, p in violations if t not in (3,)]
        penalty_violations = [(m, t, p) for m, t, p in violations if t in (3,)]

        # Increase penalty for v_min violations without splitting
        for m, t, _ in penalty_violations:
            if t == 3:
                rho_vmin_list[m] *= 2.0
                print(f"  vmin penalty boost piece {m}: rho_vmin={rho_vmin_list[m]:.0f}")

        # If only v_min violations remain, re-run with higher penalty (no split)
        if len(split_violations) == 0:
            continue

        if len(split_violations) > 1 and split_violations[0][0] == last_split:
            piece_idx, violation_type, viol_pos = split_violations[1]
        else:
            piece_idx, violation_type, viol_pos = split_violations[0]
        last_split = piece_idx

        print(piece_idx, violation_type)

        q_prev = waypoints[piece_idx]
        q_next = waypoints[piece_idx + 1]

        poly_0, poly_1, new_waypoint = split_hpoly(hpolys[piece_idx], q_prev, q_next, 0.5, split_point=viol_pos)

        waypoints.insert(piece_idx + 1, new_waypoint)
        d_prev = np.linalg.norm(new_waypoint - q_prev)
        d_next = np.linalg.norm(q_next - new_waypoint)
        ratio = d_prev / max(d_prev + d_next, 1e-10)
        T_orig = Ts[piece_idx]
        Ts[piece_idx] = T_orig * ratio
        Ts.insert(piece_idx + 1, T_orig * (1 - ratio))

        default_rho = 128.0

        if violation_type == 0:
            pakka_list.insert(piece_idx + 1, pakka_list[piece_idx])
            pakka_list[piece_idx] = pakka_list[piece_idx] * 1.5
            rho_v_list.insert(piece_idx + 1, default_rho)
            rho_a_list.insert(piece_idx + 1, default_rho)
            rho_vmin_list.insert(piece_idx + 1, default_rho)
            rho_turn_list.insert(piece_idx + 1, default_rho)
            rho_climb_list.insert(piece_idx + 1, default_rho)
        elif violation_type == 1:
            rho_v_list.insert(piece_idx + 1, rho_v_list[piece_idx])
            rho_v_list[piece_idx] = rho_v_list[piece_idx] * 1.5
            pakka_list.insert(piece_idx + 1, 1.0)
            rho_a_list.insert(piece_idx + 1, default_rho)
            rho_vmin_list.insert(piece_idx + 1, default_rho)
            rho_turn_list.insert(piece_idx + 1, default_rho)
            rho_climb_list.insert(piece_idx + 1, default_rho)
        elif violation_type == 2:
            rho_a_list.insert(piece_idx + 1, rho_a_list[piece_idx])
            rho_a_list[piece_idx] = rho_a_list[piece_idx] * 1.5
            pakka_list.insert(piece_idx + 1, 1.0)
            rho_v_list.insert(piece_idx + 1, default_rho)
            rho_vmin_list.insert(piece_idx + 1, default_rho)
            rho_turn_list.insert(piece_idx + 1, default_rho)
            rho_climb_list.insert(piece_idx + 1, default_rho)
        elif violation_type == 4:
            rho_turn_list.insert(piece_idx + 1, rho_turn_list[piece_idx])
            rho_turn_list[piece_idx] = rho_turn_list[piece_idx] * 1.5
            pakka_list.insert(piece_idx + 1, 1.0)
            rho_v_list.insert(piece_idx + 1, default_rho)
            rho_a_list.insert(piece_idx + 1, default_rho)
            rho_vmin_list.insert(piece_idx + 1, default_rho)
            rho_climb_list.insert(piece_idx + 1, default_rho)
        elif violation_type == 5:
            rho_climb_list.insert(piece_idx + 1, rho_climb_list[piece_idx])
            rho_climb_list[piece_idx] = rho_climb_list[piece_idx] * 1.5
            pakka_list.insert(piece_idx + 1, 1.0)
            rho_v_list.insert(piece_idx + 1, default_rho)
            rho_a_list.insert(piece_idx + 1, default_rho)
            rho_vmin_list.insert(piece_idx + 1, default_rho)
            rho_turn_list.insert(piece_idx + 1, default_rho)

        hpolys[piece_idx] = poly_1
        hpolys.insert(piece_idx + 1, poly_0)
    else:
        print("iter bitti")

    return opt, Ts, waypoints, hpolys
