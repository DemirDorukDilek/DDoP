import numpy as np
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class ConvexObstacle:
    vertices: np.ndarray

    def get_edges(self):
        n = len(self.vertices)
        return [(self.vertices[i], self.vertices[(i + 1) % n]) for i in range(n)]


@dataclass
class HPolyhedron:
    A: np.ndarray
    b: np.ndarray

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


def closest_point_on_vector(p, a, b):
    ab = b - a
    t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-15)
    return a + np.clip(t, 0.0, 1.0) * ab


def closest_point_on_obstacle(point, obstacle):
    best_dist, best_pt = np.inf, None
    for a, b in obstacle.get_edges():
        cp_pt = closest_point_on_vector(point, a, b)
        d = np.linalg.norm(cp_pt - point)
        if d < best_dist:
            best_dist, best_pt = d, cp_pt
    return best_pt


def inflate_obstacle(obstacle, margin):
    center = np.mean(obstacle.vertices, axis=0)
    inflated = []
    for v in obstacle.vertices:
        d = v - center
        n = np.linalg.norm(d)
        inflated.append(v + d / n * margin if n > 1e-10 else v)
    return ConvexObstacle(np.array(inflated))

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
