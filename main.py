import os
if not os.path.exists("git/main.py"):
    !rm -rfd sample_data
    !rm -rfd git
    !git clone --single-branch --branch Util https://github.com/DemirDorukDilek/DDoP git
    import sys

    top_level_container_dir = "git/"

    if top_level_container_dir not in sys.path:
        sys.path.insert(0, top_level_container_dir)

from Utils import visualize_trajectory_2d,polygon_to_polyhedron
from DDoP import DDoP



import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
import scipy.special

class DDoPnD:

    def __init__(self,Ts,waypoints,fix_times=False,fix_waypoints=False,S=4):
        self.Ts = Ts
        self.waypoints = waypoints
        self.fix_times = fix_times
        self.fix_waypoints = fix_waypoints
        self.S = S
        self.dims = [np.array(dim) for dim in zip(*waypoints)]
        self.dim = len(self.dims)
        self.M = len(Ts)
        self.opt = [DDoP(Ts.copy(),dim,True,True,S) for dim in self.dims]


        self.rho_t = 32.0
        self.rho_v = 128.0
        self.rho_a = 128.0
        self.v_max = 4.0
        self.a_max = 5.0

        self.pakka = 1.0

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
        """v_i = (q_{i+1} - q_{i-1}) / (T_i + T_{i+1})"""
        q_prev = self.waypoints[i - 1]
        q_next = self.waypoints[i + 1]
        T_sum = self.Ts[i - 1] + self.Ts[i]
        return (q_next - q_prev) / T_sum

    def _compute_a(self, i):
        """a_i = 2 * (v_out - v_in) / (T_i + T_{i+1})"""
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
        """
        v_i = (q_{i+1} - q_{i-1}) / (T_{i-1} + T_i)
        ||v_i||² = ||q_{i+1} - q_{i-1}||² / (T_{i-1} + T_i)²
        
        J_v = ρ_v * Σ g(||v_i||² - v_max²)
        """
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
                grad_T[i - 1] += self.rho_v * gp * dv_sq_dT
                grad_T[i] += self.rho_v * gp * dv_sq_dT
                
                # ∂||v||²/∂q_{i-1} = -2 * diff / T_sum²
                # ∂||v||²/∂q_{i+1} = +2 * diff / T_sum²
                dv_sq_dq = 2.0 * diff / T_sum_sq
                
                # if q_{i-1} inner point
                if i - 1 >= 1:
                    grad_q[:, i - 2] += self.rho_v * gp * (-dv_sq_dq)
                
                # if q_{i+1} inner point
                if i + 1 <= self.M - 1:
                    grad_q[:, i] += self.rho_v * gp * dv_sq_dq
        
        return grad_T, grad_q

    def _compute_gradJ_D_a(self):
        """
        a_i = 2 * (v_out - v_in) / T_sum
        where:
            v_in = (q_i - q_{i-1}) / T_{i-1}
            v_out = (q_{i+1} - q_i) / T_i
            T_sum = T_{i-1} + T_i
        """
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
                da_dT_next = -2.0 * (v_out - v_in) / T_sum_sq - 2.0 * (q_next - q_curr) / (T_sum * np.power(T_next, 2))
                
                # ∂||a||²/∂T = 2 * a · ∂a/∂T
                grad_T[i - 1] += self.rho_a * gp * 2.0 * np.dot(a, da_dT_prev)
                grad_T[i] += self.rho_a * gp * 2.0 * np.dot(a, da_dT_next)
                
                # ∂a/∂q_{i-1} = 2/(T_sum * T_prev)
                da_dq_prev = 2.0 / (T_sum * T_prev)
                
                # ∂a/∂q_i = -2/T_sum * (1/T_prev + 1/T_next)
                da_dq_curr = -2.0 / T_sum * (1.0 / T_prev + 1.0 / T_next)
                
                # ∂a/∂q_{i+1} = -2/(T_sum * T_next)
                da_dq_next = -2.0 / (T_sum * T_next)
                
                # ∂||a||²/∂q = 2 * a · ∂a/∂q = 2 * a * scalar
                # q_{i-1}
                if i - 1 >= 1:
                    grad_q[:, i - 2] += self.rho_a * gp * 2.0 * a * da_dq_prev
                
                # q_i
                grad_q[:, i - 1] += self.rho_a * gp * 2.0 * a * da_dq_curr
                
                # q_{i+1}
                if i + 1 <= self.M - 1:
                    grad_q[:, i] += self.rho_a * gp * 2.0 * a * da_dq_next
        
        return grad_T, grad_q

    def J_D(self):
        J_D_t = sum(self.Ts)
        J_D_v = 0
        J_D_a = 0
        for i in range(1,self.M):
            J_D_v += self.g(np.sum(np.power(self._compute_V(i),2))-np.power(self.v_max,2))
            J_D_a += self.g(np.sum(np.power(self._compute_a(i),2))-np.power(self.a_max,2))
        return self.rho_t*J_D_t + self.rho_v*J_D_v + self.rho_a

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
                A_j,b_j = self.polyhedra[j]
                slack = b_j - A_j @ q_i
                if np.any((slack)<=0):
                    return float("inf")
                J_F = self.pakka * np.sum(np.log(slack))
        return J_F
    
    def grad_J_F(self):
        grad_q = np.zeros((self.dim, self.M - 1))
        for i in range(1,self.M):
            q_i = self.waypoints[i]
            grad_q_i = np.zeros(self.dim)
            for j in (i-1, i):
                A_j,b_j = self.polyhedra[j]
                slack = b_j - A_j @ q_i
                if np.any((slack)<=0):
                    grad_q_i += 1e10*np.ones(self.dim)
                else:
                    grad_q_i +=  self.pakka * A_j.T @ (1.0/slack)
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
            options={'disp': True, 'maxiter': 100}
        )
        self._unpack(result.x)

        return self.Ts, self.waypoints, result.fun


waypoints = [
    (0, 0), # q₀: Başlangıç
    (1.5, 0.5), # q₁: İlk dönüş
    (2.5, 1.5), # q₂: Engeli geç
    (3.5, 0.5), # q₃: İkinci dönüş
    (5, 1), # q₄: Bitiş
]

# Polyhedra: Her piece için güvenli bölge (CCW sıralı köşeler)
polyhedra = [
    # P₀: Başlangıç bölgesi (geniş alan)
    polygon_to_polyhedron([
        (-0.5, -0.5),
        (2, -0.5),
        (2, 1.2),
        (-0.5, 1.2)
    ]),
    
    # P₁: Dar geçit (engelin solundan)
    polygon_to_polyhedron([
        (1, 0),
        (3, 0),
        (3.2, 2),
        (1.2, 2)
    ]),
    
    # P₂: Engel üstü bölge
    polygon_to_polyhedron([
        (2, 1),
        (4, 0.5),
        (4.2, 2.2),
        (2, 2.2)
    ]),
    
    # P₃: Bitiş bölgesi
    polygon_to_polyhedron([
        (3, -0.3),
        (5.5, -0.3),
        (5.5, 1.8),
        (3, 1.8)
    ]),
]

# Engeller (sadece görselleştirme için)
obstacles = [
    {'center': (2.5, 0.3), 'radius': 0.4},
    {'center': (1.8, 1.8), 'radius': 0.3},
]

opt = DDoPnD([1.0]*(len(waypoints)-1),waypoints,False,True,3)
T,way,C = opt.run()
print(T)
print(way)
print(C)

ww =[np.array(dim) for dim in zip(*way)]
visualize_trajectory_2d(opt,T,*ww)