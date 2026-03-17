from .DDoPnDm import DDoPnD

import numpy as np
from scipy.optimize import minimize


class DDoPnDFixedWing(DDoPnD):

    def __init__(self, Ts, waypoints, polyhedron,
                 rho_t=32.0, rho_v=None, rho_a=None, pakka=None,
                 fix_times=False, fix_waypoints=False, S=3,
                 v_min=10.0, phi_max_deg=35.0,
                 gamma_max_deg=20.0, gamma_min_deg=-30.0,
                 rho_vmin=None, rho_turn=None, rho_climb=None,
                 v_init=None, v_final=None,
                 **opt_args):

        opt_args.setdefault('v_max', 25.0)
        opt_args.setdefault('a_max', 5.0)

        super().__init__(Ts, waypoints, polyhedron,
                         rho_t, rho_v, rho_a, pakka,
                         fix_times, fix_waypoints, S, **opt_args)

        self.v_min = v_min
        self.phi_max = np.radians(phi_max_deg)
        self.gamma_max = np.radians(gamma_max_deg)
        self.gamma_min = np.radians(gamma_min_deg)
        self.g_acc = 9.81
        self.T_min = opt_args.get('T_min', 0.1)

        self.a_lat_max = self.g_acc * np.tan(self.phi_max)
        self.tan2_gamma_max = np.tan(self.gamma_max) ** 2
        self.tan2_gamma_min = np.tan(abs(self.gamma_min)) ** 2

        self.rho_vmin = np.array([128.0] * self.M) if rho_vmin is None else rho_vmin
        self.rho_turn = np.array([128.0] * self.M) if rho_turn is None else rho_turn
        self.rho_climb = np.array([128.0] * self.M) if rho_climb is None else rho_climb

        # Boundary velocities for fixed-wing (can't have zero velocity)
        self._setup_boundary_velocities(v_init, v_final)

        self.save_fw = (Ts.copy(), [w.copy() for w in waypoints], [p for p in polyhedron],
                        rho_t, rho_v, rho_a, pakka,
                        fix_times, fix_waypoints, S,
                        v_min, phi_max_deg, gamma_max_deg, gamma_min_deg,
                        rho_vmin, rho_turn, rho_climb,
                        v_init, v_final)
        self.save_fw_opt_args = opt_args.copy()

    def _setup_boundary_velocities(self, v_init, v_final):
        wp = self.waypoints

        if v_init is None:
            direction = np.array(wp[1]) - np.array(wp[0])
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                v_init = direction / norm * self.v_min
            else:
                v_init = np.zeros(self.dim)
                v_init[0] = self.v_min

        if v_final is None:
            direction = np.array(wp[-1]) - np.array(wp[-2])
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                v_final = direction / norm * self.v_min
            else:
                v_final = np.zeros(self.dim)
                v_final[0] = self.v_min

        self.v_init = np.array(v_init)
        self.v_final = np.array(v_final)

        # Update each sub-optimizer's boundary conditions
        for d, opt in enumerate(self.opt):
            # kappa[0] = [pos, vel, acc, ...] at start
            opt.kappa[0] = np.zeros(self.S)
            opt.kappa[0][0] = self.dims[d][0]
            opt.kappa[0][1] = self.v_init[d]

            # kappa[-1] = [pos, vel, acc, ...] at end
            opt.kappa[-1] = np.zeros(self.S)
            opt.kappa[-1][0] = self.dims[d][-1]
            opt.kappa[-1][1] = self.v_final[d]

            # Update dconst
            opt.dconst[0 * self.S + 1, 0] = self.v_init[d]
            opt.dconst[-1 * self.S + 1, 0] = self.v_final[d]

    def reset(self):
        args = self.save_fw
        self.__init__(args[0], args[1], args[2],
                      args[3], args[4], args[5], args[6],
                      args[7], args[8], args[9],
                      args[10], args[11], args[12], args[13],
                      args[14], args[15], args[16],
                      args[17], args[18],
                      **self.save_fw_opt_args)

    def _update_sub_optimizers(self):
        super()._update_sub_optimizers()
        # Re-apply boundary velocities after parent updates
        for d, opt in enumerate(self.opt):
            opt.kappa[0] = np.zeros(self.S)
            opt.kappa[0][0] = self.dims[d][0]
            opt.kappa[0][1] = self.v_init[d]

            opt.kappa[-1] = np.zeros(self.S)
            opt.kappa[-1][0] = self.dims[d][-1]
            opt.kappa[-1][1] = self.v_final[d]

            opt.dconst[0 * self.S + 1, 0] = self.v_init[d]
            opt.dconst[-1 * self.S + 1, 0] = self.v_final[d]

    def run(self):
        x0 = []
        bounds = []

        if not self.fix_times:
            x0.extend(self.Ts)
            bounds.extend([(self.T_min, None)] * self.M)

        if not self.fix_waypoints:
            for dim in self.dims:
                x0.extend(dim[1:-1])
            bounds.extend([(None, None)] * self.dim * (self.M - 1))

        x0 = np.array(x0)

        result = minimize(
            self.J, x0,
            method='L-BFGS-B',
            jac=self.grad,
            bounds=bounds,
            options={'disp': True, 'maxiter': self.opt_args.get("maxiter", 1000)}
        )
        self._unpack(result.x)
        self.J(result.x)
        return self.Ts, self.waypoints, result.fun

    # ======================== MINIMUM SPEED ========================

    def _compute_J_D_vmin(self):
        cost = 0.0
        for i in range(1, self.M):
            v = self._compute_V(i)
            v_sq = np.sum(v ** 2)
            cost += self.rho_vmin[i - 1] * self.g(self.v_min ** 2 - v_sq)
        return cost

    def _compute_gradJ_D_vmin(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        for i in range(1, self.M):
            q_prev = self.waypoints[i - 1]
            q_next = self.waypoints[i + 1]
            T_sum = self.Ts[i - 1] + self.Ts[i]
            T_sum_sq = T_sum ** 2

            diff = q_next - q_prev
            diff_sq = np.sum(diff ** 2)
            v_sq = diff_sq / T_sum_sq

            gp = self.dgdx(self.v_min ** 2 - v_sq)

            if gp > 0:
                # d(v_min^2 - v_sq)/dT = +2*diff_sq/T_sum^3
                dv_sq_dT = -2.0 * diff_sq / (T_sum ** 3)

                grad_T[i - 1] += self.rho_vmin[i - 1] * gp * (-dv_sq_dT)
                grad_T[i] += self.rho_vmin[i - 1] * gp * (-dv_sq_dT)

                dv_sq_dq = 2.0 * diff / T_sum_sq

                if i - 1 >= 1:
                    grad_q[:, i - 2] += self.rho_vmin[i - 1] * gp * dv_sq_dq
                if i + 1 <= self.M - 1:
                    grad_q[:, i] += self.rho_vmin[i - 1] * gp * (-dv_sq_dq)

        return grad_T, grad_q

    # ======================== LATERAL ACCELERATION (TURN RADIUS) ========================

    def _compute_J_D_turn(self):
        cost = 0.0
        for i in range(1, self.M):
            v = self._compute_V(i)
            a = self._compute_a(i)
            s = np.sum(v ** 2)

            if s < 1e-10:
                continue

            r = np.sum(a ** 2)
            p = np.dot(a, v)
            a_lat_sq = r - p ** 2 / s

            cost += self.rho_turn[i - 1] * self.g(a_lat_sq - self.a_lat_max ** 2)
        return cost

    def _compute_gradJ_D_turn(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        for i in range(1, self.M):
            q_prev = self.waypoints[i - 1]
            q_curr = self.waypoints[i]
            q_next = self.waypoints[i + 1]
            T_prev = self.Ts[i - 1]
            T_next = self.Ts[i]
            T_sum = T_prev + T_next
            T_sum_sq = T_sum ** 2

            diff = q_next - q_prev
            v = diff / T_sum
            s = np.sum(v ** 2)

            if s < 1e-10:
                continue

            v_in = (q_curr - q_prev) / T_prev
            v_out = (q_next - q_curr) / T_next
            a = 2.0 * (v_out - v_in) / T_sum

            r = np.sum(a ** 2)
            p = np.dot(a, v)
            a_lat_sq = r - p ** 2 / s

            gp = self.dgdx(a_lat_sq - self.a_lat_max ** 2)

            if gp <= 0:
                continue

            dv_dT = -diff / T_sum_sq

            dv_dq_prev_scale = -1.0 / T_sum
            dv_dq_next_scale = 1.0 / T_sum

            da_dT_prev = -2.0 * (v_out - v_in) / T_sum_sq + 2.0 * (q_curr - q_prev) / (T_sum * T_prev ** 2)
            da_dT_next = -2.0 * (v_out - v_in) / T_sum_sq - 2.0 * (q_next - q_curr) / (T_sum * T_next ** 2)

            da_dq_prev_scale = 2.0 / (T_sum * T_prev)
            da_dq_curr_scale = -2.0 / T_sum * (1.0 / T_prev + 1.0 / T_next)
            da_dq_next_scale = 2.0 / (T_sum * T_next)

            # --- w.r.t. T_prev ---
            dr_dT_prev = 2.0 * np.dot(a, da_dT_prev)
            ds_dT_prev = 2.0 * np.dot(v, dv_dT)
            dp_dT_prev = np.dot(da_dT_prev, v) + np.dot(a, dv_dT)
            dalat_dT_prev = dr_dT_prev - (2.0 * p * dp_dT_prev * s - p ** 2 * ds_dT_prev) / s ** 2
            grad_T[i - 1] += self.rho_turn[i - 1] * gp * dalat_dT_prev

            # --- w.r.t. T_next ---
            dr_dT_next = 2.0 * np.dot(a, da_dT_next)
            ds_dT_next = ds_dT_prev
            dp_dT_next = np.dot(da_dT_next, v) + np.dot(a, dv_dT)
            dalat_dT_next = dr_dT_next - (2.0 * p * dp_dT_next * s - p ** 2 * ds_dT_next) / s ** 2
            grad_T[i] += self.rho_turn[i - 1] * gp * dalat_dT_next

            # --- w.r.t. q ---
            def _dalat_dq(a_scale, v_scale):
                dr = 2.0 * a * a_scale
                ds = 2.0 * v * v_scale
                dp_vec = v * a_scale + a * v_scale
                return dr - (2.0 * p * dp_vec * s - p ** 2 * ds) / s ** 2

            if i - 1 >= 1:
                grad_q[:, i - 2] += self.rho_turn[i - 1] * gp * _dalat_dq(da_dq_prev_scale, dv_dq_prev_scale)

            grad_q[:, i - 1] += self.rho_turn[i - 1] * gp * _dalat_dq(da_dq_curr_scale, 0.0)

            if i + 1 <= self.M - 1:
                grad_q[:, i] += self.rho_turn[i - 1] * gp * _dalat_dq(da_dq_next_scale, dv_dq_next_scale)

        return grad_T, grad_q

    # ======================== FLIGHT PATH ANGLE (3D ONLY) ========================

    def _compute_J_D_climb(self):
        if self.dim != 3:
            return 0.0

        cost = 0.0
        for i in range(1, self.M):
            v = self._compute_V(i)
            vz = v[2]
            vh_sq = v[0] ** 2 + v[1] ** 2

            if vz > 0:
                cost += self.rho_climb[i - 1] * self.g(vz ** 2 - vh_sq * self.tan2_gamma_max)
            elif vz < 0:
                cost += self.rho_climb[i - 1] * self.g(vz ** 2 - vh_sq * self.tan2_gamma_min)
        return cost

    def _compute_gradJ_D_climb(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        if self.dim != 3:
            return grad_T, grad_q

        for i in range(1, self.M):
            q_prev = self.waypoints[i - 1]
            q_next = self.waypoints[i + 1]
            T_sum = self.Ts[i - 1] + self.Ts[i]
            T_sum_sq = T_sum ** 2

            diff = q_next - q_prev
            v = diff / T_sum
            vz = v[2]
            vh_sq = v[0] ** 2 + v[1] ** 2

            if vz > 0:
                tan2 = self.tan2_gamma_max
            elif vz < 0:
                tan2 = self.tan2_gamma_min
            else:
                continue

            arg = vz ** 2 - vh_sq * tan2
            gp = self.dgdx(arg)

            if gp <= 0:
                continue

            dh_dv = np.array([-2.0 * v[0] * tan2,
                              -2.0 * v[1] * tan2,
                              2.0 * vz])

            dv_dT = -diff / T_sum_sq

            dh_dT = np.dot(dh_dv, dv_dT)
            grad_T[i - 1] += self.rho_climb[i - 1] * gp * dh_dT
            grad_T[i] += self.rho_climb[i - 1] * gp * dh_dT

            dh_dq_prev = dh_dv * (-1.0 / T_sum)
            dh_dq_next = dh_dv * (1.0 / T_sum)

            if i - 1 >= 1:
                grad_q[:, i - 2] += self.rho_climb[i - 1] * gp * dh_dq_prev
            if i + 1 <= self.M - 1:
                grad_q[:, i] += self.rho_climb[i - 1] * gp * dh_dq_next

        return grad_T, grad_q

    # ======================== OVERRIDES ========================

    def J_D(self):
        cost = super().J_D()
        cost += self._compute_J_D_vmin()
        cost += self._compute_J_D_turn()
        cost += self._compute_J_D_climb()
        return cost

    def grad_J_D(self):
        grad_T, grad_q = super().grad_J_D()

        gT_vmin, gq_vmin = self._compute_gradJ_D_vmin()
        grad_T += gT_vmin
        grad_q += gq_vmin

        gT_turn, gq_turn = self._compute_gradJ_D_turn()
        grad_T += gT_turn
        grad_q += gq_turn

        gT_climb, gq_climb = self._compute_gradJ_D_climb()
        grad_T += gT_climb
        grad_q += gq_climb

        return grad_T, grad_q
