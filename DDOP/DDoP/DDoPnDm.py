from .DDoPm import DDoP

import numpy as np
from scipy.optimize import minimize


class DDoPnD:

    def __init__(self,Ts,waypoints,polyhedron,rho_t=None,rho_v=None,rho_a=None,pakka=None,fix_times=False,fix_waypoints=False,S=4):
        self.save = Ts.copy(),waypoints.copy(),polyhedron.copy()
        self.Ts = Ts
        self.waypoints = waypoints
        self.fix_times = fix_times
        self.fix_waypoints = fix_waypoints
        self.S = S
        self.dims = [np.array(dim) for dim in zip(*waypoints)]
        self.dim = len(self.dims)
        self.M = len(Ts)
        self.opt = [DDoP(Ts.copy(),dim,True,True,S) for dim in self.dims]
        self.polyhedron = polyhedron

        self.rho_t = 32.0 if rho_t == None else rho_t
        self.rho_v = np.array([128.0]*self.M) if rho_v is None else rho_v
        self.rho_a = np.array([128.0]*self.M) if rho_a is None else rho_a
        self.v_max = 13.0
        self.a_max = 10.0

        self.pakka = np.array([1.0]*self.M) if pakka is None else pakka

    def reset(self):
        self.__init__(*self.save,self.fix_times,self.fix_waypoints,self.S)

    def _unpack(self,x):
        idx = 0
        if not self.fix_times:
            self.Ts = x[idx:idx+self.M]
            idx += self.M
        if not self.fix_waypoints:
            n_inner = self.M-1
            for i in range(self.dim):
                self.dims[i] = np.concatenate([[self.dims[i][0]], x[idx:idx+n_inner], [self.dims[i][-1]]])
                idx = idx+n_inner
            self.waypoints = np.column_stack(self.dims)



    def _update_sub_optimizers(self):
        for idx,opt in enumerate(self.opt):
            opt.Ts = self.Ts.copy()
            opt.checkpoints = self.dims[idx].copy()
            opt._update_dconst_kappa()

    def g(self,x):
        return np.power(max(x,0),3)
    def dgdx(self,x):
        return 3*np.power(max(x,0),2)
    def _compute_V(self, i):
        q_prev = self.waypoints[i - 1]
        q_next = self.waypoints[i + 1]
        T_sum = self.Ts[i - 1] + self.Ts[i]
        return (q_next - q_prev) / T_sum

    def _compute_a(self, i):
        q_prev = self.waypoints[i - 1]
        q_curr = self.waypoints[i]
        q_next = self.waypoints[i + 1]
        T_prev = self.Ts[i - 1]
        T_next = self.Ts[i]

        v_in = (q_curr - q_prev) / T_prev
        v_out = (q_next - q_curr) / T_next
        T_sum = T_prev + T_next

        return 2.0 * (v_out - v_in) / T_sum

    def _compute_gradJ_D_v(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        for i in range(1, self.M):
            q_prev = self.waypoints[i - 1]
            q_curr = self.waypoints[i]
            q_next = self.waypoints[i + 1]
            T_sum = self.Ts[i - 1] + self.Ts[i]
            T_sum_sq = np.power(T_sum,2)

            diff = q_next - q_prev
            diff_sq = np.sum(np.power(diff,2))
            v_sq = diff_sq /T_sum_sq

            gp = self.dgdx(v_sq - np.power(self.v_max, 2))

            if gp > 0:
                # ∂||v||²/∂T = -2 * ||diff||² / T_sum³
                dv_sq_dT = -2.0 * diff_sq / np.power(T_sum,3)

                # ∂J_v/∂T_{i-1} and ∂J_v/∂T_i
                grad_T[i - 1] += self.rho_v[i-1] * gp * dv_sq_dT
                grad_T[i] += self.rho_v[i-1] * gp * dv_sq_dT

                # ∂||v||²/∂q_{i-1} = -2 * diff / T_sum²
                # ∂||v||²/∂q_{i+1} = +2 * diff / T_sum²
                dv_sq_dq = 2.0 * diff / T_sum_sq

                # if q_{i-1} inner point
                if i - 1 >= 1:
                    grad_q[:, i - 2] += self.rho_v[i-1] * gp * (-dv_sq_dq)

                # if q_{i+1} inner point
                if i + 1 <= self.M - 1:
                    grad_q[:, i] += self.rho_v[i-1] * gp * dv_sq_dq

        return grad_T, grad_q

    def _compute_gradJ_D_a(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        for i in range(1, self.M):
            q_prev = self.waypoints[i - 1]
            q_curr = self.waypoints[i]
            q_next = self.waypoints[i + 1]
            T_prev = self.Ts[i - 1]
            T_next = self.Ts[i]
            T_sum = T_prev + T_next
            T_sum_sq = np.power(T_sum,2)

            v_in = (q_curr - q_prev) / T_prev
            v_out = (q_next - q_curr) / T_next
            a = 2.0 * (v_out - v_in) / T_sum
            a_sq = np.sum(np.power(a, 2))

            gp = self.dgdx(a_sq - self.a_max ** 2)

            if gp > 0:
                # ∂a/∂T_{i-1} (T_prev)
                # a = 2/T_sum * (v_out - v_in)
                # ∂a/∂T_prev = -2/T_sum² * (v_out - v_in) + 2/T_sum * (q_curr - q_prev)/T_prev²
                da_dT_prev = -2.0 * (v_out - v_in) / T_sum_sq + 2.0 * (q_curr - q_prev) / (T_sum * np.power(T_prev, 2))

                # ∂a/∂T_i (T_next)
                da_dT_next = 2.0 * (v_out - v_in) / T_sum_sq - 2.0 * (q_next - q_curr) / (T_sum * np.power(T_next, 2))

                # ∂||a||²/∂T = 2 * a · ∂a/∂T
                grad_T[i - 1] += self.rho_a[i-1] * gp * 2.0 * np.dot(a, da_dT_prev)
                grad_T[i] += self.rho_a[i-1] * gp * 2.0 * np.dot(a, da_dT_next)

                # ∂a/∂q_{i-1} = 2/(T_sum * T_prev)
                da_dq_prev = 2.0 / (T_sum * T_prev)

                # ∂a/∂q_i = -2/T_sum * (1/T_prev + 1/T_next)
                da_dq_curr = -2.0 / T_sum * (1.0 / T_prev + 1.0 / T_next)

                # ∂a/∂q_{i+1} = -2/(T_sum * T_next)
                da_dq_next = -2.0 / (T_sum * T_next)

                # ∂||a||²/∂q = 2 * a · ∂a/∂q = 2 * a * scalar
                # q_{i-1}
                if i - 1 >= 1:
                    grad_q[:, i - 2] += self.rho_a[i-1] * gp * 2.0 * a * da_dq_prev

                # q_i
                grad_q[:, i - 1] += self.rho_a[i-1] * gp * 2.0 * a * da_dq_curr

                # q_{i+1}
                if i + 1 <= self.M - 1:
                    grad_q[:, i] += self.rho_a[i-1] * gp * 2.0 * a * da_dq_next

        return grad_T, grad_q

    def J_D(self):
        J_D_t = sum(self.Ts)
        J_D_v = 0
        J_D_a = 0
        for i in range(1,self.M):
            J_D_v += self.rho_v[i-1]*self.g(np.sum(np.power(self._compute_V(i),2))-np.power(self.v_max,2))
            J_D_a += self.rho_a[i-1]*self.g(np.sum(np.power(self._compute_a(i),2))-np.power(self.a_max,2))
        return self.rho_t*J_D_t + J_D_v + J_D_a

    def grad_J_D(self):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        grad_T += self.rho_t

        grad_T_v, grad_q_v = self._compute_gradJ_D_v()
        grad_T += grad_T_v
        grad_q += grad_q_v

        grad_T_a, grad_q_a = self._compute_gradJ_D_a()
        grad_T += grad_T_a
        grad_q += grad_q_a

        return grad_T,grad_q

    def J_F(self):
        J_F = 0
        for i in range(1,self.M):
            q_i = self.waypoints[i]
            for j in (i-1, i):
                A_j,b_j = self.polyhedron[j]
                slack = b_j - A_j @ q_i
                if np.any((slack)<=0):
                    # print("inf",q_i,b_j,A_j)
                    return 1e10

                J_F -= self.pakka[i-1] * np.sum(np.log(slack))
        return J_F

    def grad_J_F(self):
        grad_q = np.zeros((self.dim, self.M - 1))
        for i in range(1,self.M):
            q_i = self.waypoints[i]
            grad_q_i = np.zeros(self.dim)
            for j in (i-1, i):
                A_j,b_j = self.polyhedron[j]
                slack = b_j - A_j @ q_i
                if np.any((slack)<=0):
                    # print("grad_inf")
                    grad_q_i += 1e10*np.ones(self.dim)
                else:
                    grad_q_i +=  self.pakka[i-1] * (A_j.T @ (1.0/slack))
            grad_q[:,i-1] = grad_q_i
        return grad_q


    def J(self, x):
        self._unpack(x)
        self._update_sub_optimizers()

        cost = 0.0
        for opt in self.opt:
            cost+= opt.J(np.array([]))

        cost += self.J_D()
        cost += self.J_F()
        return float(cost)

    def grad(self, x):
        self._unpack(x)
        self._update_sub_optimizers()
        grad_T = np.zeros(self.M)
        grad_q = np.zeros((self.dim, self.M - 1))

        for opt_idx,opt in enumerate(self.opt):
            for i in range(self.M):
                grad_T[i] += opt.compute_dJdT(i).item()

            for i in range(self.M - 1):
                grad_q[opt_idx, i] += opt.compute_dJdq(i).item()

        grad_J_D_T,grad_J_D_q = self.grad_J_D()
        grad_T += grad_J_D_T
        grad_q += grad_J_D_q

        grad_q += self.grad_J_F()
        grad = []
        if not self.fix_times:
            grad.extend(grad_T)
        if not self.fix_waypoints:
            for d in range(self.dim):
                grad.extend(grad_q[d])
        return np.array(grad)

    def run(self):
        x0 = []
        bounds = []

        if not self.fix_times:
            x0.extend(self.Ts)
            bounds.extend([(0.01, None)] * self.M)

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
            options={'disp': True, 'maxiter': 1000}
        )
        self._unpack(result.x)

        return self.Ts, self.waypoints, result.fun
