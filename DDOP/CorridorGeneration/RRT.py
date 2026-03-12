import numpy as np
from dataclasses import dataclass
from typing import Optional
from .Corridor_Utils import inflate_obstacle, HPolyhedron


@dataclass
class RRTNode:
    pos: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0


def segment_intersects_hpoly(a, b, hpoly: HPolyhedron):
    """
    Cyrus-Beck: p(t) = a + t*(b-a), t in [0,1]
    A @ p(t) <= b_poly => A @ a + t * A @ d <= b_poly
    Her halfspace icin t araligini daralt.
    Kesisim varsa True doner.
    """
    d = b - a
    Ad = hpoly.A @ d
    Aa = hpoly.A @ a

    t_min, t_max = 0.0, 1.0

    for i in range(len(hpoly.b)):
        num = hpoly.b[i] - Aa[i]
        den = Ad[i]

        if abs(den) < 1e-15:
            # Ray halfspace'e paralel
            if num < -1e-10:
                return False
            continue

        t = num / den
        if den < 0:
            # Giren: t_min yukari
            t_min = max(t_min, t)
        else:
            # Cikan: t_max asagi
            t_max = min(t_max, t)

        if t_min > t_max + 1e-10:
            return False

    return t_min <= t_max + 1e-10


def collision_free(a, b, obstacles):
    for obs in obstacles:
        if segment_intersects_hpoly(a, b, obs):
            return False
    return True


def point_in_free_space(point, obstacles):
    return not any(obs.contains(point) for obs in obstacles)


def steer(from_pos, to_pos, step_size):
    diff = to_pos - from_pos
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return to_pos.copy()
    return from_pos + (diff / dist) * step_size


def rrt(start, goal, obstacles, bounds,
             max_iter=3000, step_size=0.5, goal_bias=0.1,
             goal_tol=0.5, safety_margin=0.1, **_):
    dim = len(start)
    lower, upper = bounds

    if safety_margin > 0:
        check_obs = [inflate_obstacle(o, safety_margin) for o in obstacles]
    else:
        check_obs = obstacles

    nodes = [RRTNode(pos=start.copy())]
    goal_idx = None

    for iter in range(max_iter):
        sample = goal.copy() if np.random.rand() < goal_bias else \
                 lower + np.random.rand(dim) * (upper - lower)

        dists = [np.linalg.norm(n.pos - sample) for n in nodes]
        nearest_idx = int(np.argmin(dists))
        nearest = nodes[nearest_idx]
        new_pos = steer(nearest.pos, sample, step_size)

        if not point_in_free_space(new_pos, check_obs):
            continue
        if not collision_free(nearest.pos, new_pos, check_obs):
            continue

        new_idx = len(nodes)
        nodes.append(RRTNode(pos=new_pos, parent=nearest_idx))

        d2g = np.linalg.norm(new_pos - goal)
        if d2g < goal_tol and collision_free(new_pos, goal, check_obs):
            goal_idx = len(nodes)
            nodes.append(RRTNode(pos=goal.copy(), parent=new_idx))
            break
    else:
        print("AKAME",_)

    if goal_idx is None:
        return None

    path = []
    idx = goal_idx
    while idx is not None:
        path.append(nodes[idx].pos)
        idx = nodes[idx].parent
    path.reverse()
    return np.array(path)