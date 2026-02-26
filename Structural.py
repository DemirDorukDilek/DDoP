import numpy as np
import matplotlib as plt
import scipy
from Structural import *

def compute_As(T,S):
    f = scipy.special.factorial(np.arange(S)).T
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    jmi = j - i
    njmi = jmi.copy()
    njmi[njmi < 0] = 0
    j_fact = scipy.special.factorial(j)

    O = np.zeros((S, S))
    E = np.eye(S)*f
    U = np.eye(S)*1/f
    F = np.zeros((S, S))
    G = scipy.special.factorial(S+j)/scipy.special.factorial(S+j-i)*np.power(T,S+j-i)

    cond = i<=j
    F[cond] = np.where(cond,j_fact,0)[cond]/np.where(cond,scipy.special.factorial(j-i),0)[cond]*np.where(cond,np.power(T,njmi),0)[cond]


    I = np.broadcast_to(i+1, (S, S))
    J = np.broadcast_to(j+1, (S, S))
    K = np.arange(S + 1)[None, None, :]
    upper = S - np.maximum(I, J)
    mask_k = (K <= upper[..., None])
    n1 = I[..., None] + K
    n2 = (2*S - J[..., None] - K - 1)

    mask = (mask_k & (n1 >= 0) & (n1 <= S) & (n2 >= 0) & (n2 >= (S-1)))
    term = ((-1.0) ** K) * scipy.special.comb(S, n1, exact=False) * scipy.special.comb(n2, S-1, exact=False)
    v = np.sum(np.where(mask, term, 0.0), axis=-1)
    V = v/(j_fact*np.where(i%2, 1 ,-1)*np.power(T,S-J+I))


    n1 = I[..., None] - 1
    mask = (mask_k & (n1 >= 0) & (n1 <= S) & (n2 >= 0) & (n2 >= (S-1)))
    term = scipy.special.comb(S-K-1, n1, exact=False) * scipy.special.comb(n2, S-1, exact=False)
    w = np.sum(np.where(mask, term, 0.0), axis=-1)
    W = w/(j_fact*np.where((i+j)%2, -1 ,1)*np.power(T,S-J+I))

    Af = np.block([[E, O],[F,G]])
    Ab = np.block([[U, O],[V,W]])
    return Af,Ab

def compute_Q(T,N,S):
    i = (np.arange(N+1-S)+S)[:, None]
    j = (np.arange(N+1-S)+S)[None , :]
    temp = i+j-2*S+1
    Qbr = np.power(T,temp)/temp*scipy.special.factorial(i)/scipy.special.factorial(i-S)*scipy.special.factorial(j)/scipy.special.factorial(j-S)
    Q = np.zeros((N+1,N+1))
    Q[N-S+1:,N-S+1:] = Qbr
    return Q

def compute_dQdT(T,N,S):
    i = (np.arange(N+1-S)+S)[:, None]
    j = (np.arange(N+1-S)+S)[None , :]
    temp = i+j-2*S
    dQdTbr = np.power(T,temp)*scipy.special.factorial(i)/scipy.special.factorial(i-S)*scipy.special.factorial(j)/scipy.special.factorial(j-S)
    dQdT = np.zeros((N+1,N+1))
    dQdT[N-S+1:,N-S+1:] = dQdTbr
    return dQdT

def compute_AbDot(T,V,W,S):
    i = np.arange(S)[:, None]
    j = np.arange(S)[None, :]
    O = np.zeros((S, S))
    I = np.broadcast_to(i+1, (S, S))
    J = np.broadcast_to(j+1, (S, S))

    dV = V*(J-S-I)/T
    dW = W*(J-S-I)/T
    AbDot = np.block([[O, O],[dV,dW]])
    return AbDot

def BandedPLU(M, b, block_size):
    n = M.shape[0]
    lower = block_size
    upper = block_size
    ab = np.zeros((lower + upper + 1, n))

    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            ab[upper + i - j, j] = M[i, j]

    x = scipy.linalg.solve_banded((lower, upper), ab, b)
    return x