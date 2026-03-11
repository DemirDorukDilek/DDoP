from .DDoPnDm import DDoPnD
import numpy as np

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


def split_hpoly(hpoly, q_prev, q_next, pmargin=0.1):
    A,b = hpoly
    margin = min(pmargin, 0.25*np.linalg.norm(q_next-q_prev))

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

