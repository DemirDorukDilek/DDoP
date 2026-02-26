import numpy as np
import matplotlib as plt
import scipy

class DDoP:

    def __init__(self,Ts,checkpoints,fix_times=False,fix_waypoints=False,S=4):
        self.Ts = Ts
        self.checkpoints = checkpoints
        self.S = S
        self.N = 2 * S -1
        self.D = np.zeros((S-1,S))
        self.D[:,1:] = np.eye(S-1)
        self.M = len(Ts)
        self.dconst = np.zeros((S*len(checkpoints),1))
        for loop,i in enumerate(checkpoints):
            self.dconst[loop*S,0] = i
        self.Mm = np.zeros(((S-1)*(self.M-1),(S-1)*(self.M-1)))
        self.kappa = [np.array([x]+[0]*(S-1)) for x in checkpoints]
        self.fix_times = fix_times
        self.fix_waypoints = fix_waypoints

    def construct_block_matries(self,T):
        Af,Ab = compute_As(T,self.S)
        Q = compute_Q(T,self.N,self.S)
        H = Ab.T@Q@Ab

        GAMA = H[:self.S,:self.S]
        LAMBDA = H[:self.S,self.S:]
        PHI = H[self.S:,:self.S]
        OMEGA = H[self.S:,self.S:]
        return Af,Ab,Q,H,GAMA,LAMBDA,PHI,OMEGA

    def construct_lin_eq(self):
        self.Mm = np.zeros(self.Mm.shape)
        self.b = np.zeros(((self.S-1)*(self.M-1)))
        for i in range(self.M-1):
            if i < self.M-2:
                x = np.block([[self.D@(self.OMEGAs[i]+self.GAMAs[i+1])@self.D.T, self.D@self.LAMBDAs[i+1]@self.D.T], [self.D@self.PHIs[i+1]@self.D.T, np.zeros(((self.S-1),(self.S-1)))]])
                self.Mm[(self.S-1)*i:(self.S-1)*(i+2) , (self.S-1)*i:(self.S-1)*(i+2)] = x
            self.b[(self.S-1)*i:(self.S-1)*(i+1)] = (-self.D@(self.PHIs[i]@self.kappa[i] + (self.OMEGAs[i]+self.GAMAs[i+1])@self.kappa[i+1] + self.LAMBDAs[i]@self.kappa[i+2]))

        self.Mm[-(self.S-1):,-(self.S-1):] = self.D@(self.OMEGAs[self.M-2]+self.GAMAs[self.M-1])@self.D.T
        self.b = self.b[:,None]

    def compute_dstar(self):
        dtildastar = BandedPLU(self.Mm,self.b,(self.S-1)*2)
        # dtildastar = np.linalg.solve(self.Mm,self.b)

        B = np.zeros(((self.M+1)*self.S,(self.M-1)*(self.S-1)))
        for i in range(self.M-1):
            B[(i+1)*self.S+1:(i+1)*self.S+1+self.S-1 , (self.S-1)*i:(self.S-1)*(i+1)] = np.eye(self.S-1)

        P = np.zeros(((self.M+1)*self.S,self.M*2*self.S))
        for i in range(self.M):
            P[i*self.S:i*self.S+(2*self.S) , (2*self.S)*i:(2*self.S)*(i+1)] = np.eye(2*self.S)
        P = P.T
        self.dstar = P@(self.dconst+(B@dtildastar))

    def compute_dJdT(self,Tidx):
        T = self.Ts[Tidx]
        Q = self.Qs[Tidx]
        # Af = Afs[Tidx]
        d = self.dstar[2*self.S*Tidx:2*self.S*(Tidx+1)]
        c = self.Abs[Tidx]@d
        dQdT = compute_dQdT(T,self.N,self.S)
        AbDot = compute_AbDot(T,self.Abs[Tidx][self.S:,:self.S],self.Abs[Tidx][self.S:,self.S:],self.S)

        term1 = c.T @ dQdT @ c

        dJdc = 2 * Q @ c
        term2 = dJdc.T @ AbDot @ d # dJdc.T @ AbDot @ Af @ c aynı şey
        return term1 + term2

    def compute_dJdq(self,Tidx):
        d = self.dstar[2*self.S*Tidx:2*self.S*(Tidx+1)]
        dp1 = self.dstar[2*self.S*(Tidx+1):2*self.S*(Tidx+2)]
        c = self.Abs[Tidx]@d
        cp1 = self.Abs[Tidx+1]@dp1

        dJdc = 2 * self.Qs[Tidx] @ c
        dJdcp1 = 2 * self.Qs[Tidx+1] @ cp1

        eye = np.eye(2*self.S)

        term1 = dJdc.T @ self.Abs[Tidx] @ eye[:,self.S]
        term2 = dJdcp1.T @ self.Abs[Tidx+1] @ eye[:,0]
        return term1 + term2

    def J(self,x):
        self.__init__(self.Ts,self.checkpoints,self.fix_times,self.fix_waypoints,self.S)
        self._unpack(x)

        self.Afs,self.Abs,self.Qs,self.Hs = [],[],[],[]
        self.GAMAs,self.LAMBDAs,self.PHIs,self.OMEGAs=[],[],[],[]

        for T in self.Ts:
            Af,Ab,Q,H,GAMA,LAMBDA,PHI,OMEGA = self.construct_block_matries(T)
            self.Afs.append(Af)
            self.Abs.append(Ab)
            self.Qs.append(Q)
            self.Hs.append(H)
            self.GAMAs.append(GAMA)
            self.LAMBDAs.append(LAMBDA)
            self.PHIs.append(PHI)
            self.OMEGAs.append(OMEGA)
        self.construct_lin_eq()
        self.compute_dstar()

        cost = 0.0
        for i in range(self.M):
            d_i = self.dstar[2*self.S*i : 2*self.S*(i+1)]
            Ab = self.Abs[i]
            c_i = Ab @ d_i
            Q = self.Qs[i]
            cost += c_i.T @ Q @ c_i

        return cost.item()

    def _unpack(self, x):
        idx = 0
        if not self.fix_times:
            self.Ts = x[idx:idx+self.M]
            idx += self.M
        if not self.fix_waypoints:
            inner = x[idx:idx+self.M-1]
            self.checkpoints = np.concatenate([[self.checkpoints[0]], inner, [self.checkpoints[-1]]])
            self.kappa = [np.array([p]+[0]*(self.S-1)) for p in self.checkpoints]
            self.dconst = np.zeros((self.S*len(self.checkpoints),1))
            for loop,i in enumerate(self.checkpoints):
                self.dconst[loop*self.S,0] = i

    def _update_dconst_kappa(self):
        self.kappa = [np.array([p]+[0]*(self.S-1)) for p in self.checkpoints]
        self.dconst = np.zeros((self.S*len(self.checkpoints),1))
        for loop,i in enumerate(self.checkpoints):
            self.dconst[loop*self.S,0] = i

    def grad(self,*args):
        grad_T = np.zeros(self.M)
        grad_q = np.zeros(self.M-1)
        for i in range(self.M):
            grad_T[i] = self.compute_dJdT(i).item()
            if i < self.M-1:
                grad_q[i] = self.compute_dJdq(i).item()

        grad = []
        if not self.fix_times:
            grad.extend(grad_T)
        if not self.fix_waypoints:
            grad.extend(grad_q)
        return np.array(grad)



    def run(self):

        x0 = []
        bounds = []
        if not self.fix_times:
            bounds.extend([(0.01, None)] * self.M)
            x0.extend(self.Ts)
        if not self.fix_waypoints:
            bounds.extend([(None, None)] * (self.M-1))
            x0.extend(self.checkpoints[1:-1])

        result = minimize(
            self.J,
            x0,
            method='L-BFGS-B',
            jac=self.grad,
            bounds=bounds,
            options={'disp': True, 'maxiter': 100}
        )
        self._unpack(result.x)
        return self.Ts,self.checkpoints