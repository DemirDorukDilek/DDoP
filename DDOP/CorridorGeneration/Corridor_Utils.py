import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from scipy.spatial import ConvexHull

@dataclass
class HPolyhedron:
    A: np.ndarray
    b: np.ndarray

    @staticmethod
    def from_Vrep(vs):
        hull = ConvexHull(vs)
        A = hull.equations[:,:-1]
        b = -hull.equations[:, -1]
        return HPolyhedron(A,b)

    def contains(self, point):
        return np.all(self.A @ point <= self.b + 1e-8)

@dataclass
class Ellipsoid:
    C: np.ndarray
    d: np.ndarray

    @property
    def volume(self):
        return abs(np.linalg.det(self.C))


@dataclass
class Corridor:
    hpoly: HPolyhedron
    ellipsoid: Ellipsoid
    seed: np.ndarray


def closest_point_on_obstacle(point, obstacle):
    x = cp.Variable(len(point))
    porb = cp.Problem(cp.Minimize(cp.sum_squares(x-point)),[obstacle.A @ x <= obstacle.b])
    try:
        prob.solve(solver = cp.CLARABEL, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"]:
            return x.value
    except:
        pass
    return point.copy()


def inflate_obstacle(obstacle, margin):
    norm = np.linalg.norm(obstacle.A,axis=1)
    return HPolyhedron(obstacle.A.copy(),obstacle.b+margin*norm)

def intersect_hpoly(h1: HPolyhedron, h2: HPolyhedron):
    A = np.vstack([h1.A, h2.A])
    b = np.concatenate([h1.b, h2.b])
    return HPolyhedron(A, b)


def chebyshev_center(hpoly: HPolyhedron):
    """
    Find Chebyshev center of polytope {x : Ax <= b}.
    This is the point furthest from all walls — the "safest" interior point.

    Solves:
        maximize  r
        s.t.      aᵢᵀx + r·||aᵢ|| ≤ bᵢ,  for all i
                  r ≥ 0

    Returns center point, or None if infeasible (empty polytope).
    """
    A, b = hpoly.A, hpoly.b
    m, dim = A.shape

    x = cp.Variable(dim)
    r = cp.Variable()

    # ||aᵢ|| precomputed
    norms = np.linalg.norm(A, axis=1)

    constraints = [A[i] @ x + r * norms[i] <= b[i] for i in range(m)]
    constraints.append(r >= 0)

    prob = cp.Problem(cp.Maximize(r), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and x.value is not None:
            if r.value is not None and r.value > 1e-10:
                return x.value
    except Exception:
        pass
    return None
