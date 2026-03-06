import numpy as np
from dataclasses import dataclass
from typing import Optional
from .Corridor_Utils import inflate_obstacle


@dataclass
class RRTNode:
    pos: np.ndarray
    parent: Optional[int] = None
    cost: float = 0.0


def segments_intersect(p1, p2, p3, p4):
    d1, d2 = p2 - p1, p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-12:
        return False
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / cross
    return 0 <= t <= 1 and 0 <= u <= 1


def collision_free(a, b, obstacles):
    for obs in obstacles:
        for e1, e2 in obs.get_edges():
            if segments_intersect(a, b, e1, e2):
                return False
    return True


def point_in_obstacle(point, obs):
    verts = obs.vertices
    n = len(verts)
    for i in range(n):
        edge = verts[(i + 1) % n] - verts[i]
        to_pt = point - verts[i]
        if edge[0] * to_pt[1] - edge[1] * to_pt[0] < -1e-10:
            return False
    return True


def point_in_free_space(point, obstacles):
    return not any(point_in_obstacle(point, o) for o in obstacles)


def steer(from_pos, to_pos, step_size):
    diff = to_pos - from_pos
    dist = np.linalg.norm(diff)
    if dist <= step_size:
        return to_pos.copy()
    return from_pos + (diff / dist) * step_size


def rrt_star(start, goal, obstacles, bounds,
             max_iter=3000, step_size=0.5, goal_bias=0.1,
             goal_tol=0.5, rewire_radius=None, safety_margin=0.1):
    dim = len(start)
    lower, upper = bounds

    # Inflate obstacles for safety
    if safety_margin > 0:
        check_obs = [inflate_obstacle(o, safety_margin) for o in obstacles]
    else:
        check_obs = obstacles

    nodes = [RRTNode(pos=start.copy(), cost=0.0)]
    goal_idx = None
    best_goal_cost = np.inf
    if rewire_radius is None:
        rewire_radius = min(step_size * 3, 2.0)

    for _ in range(max_iter):
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

        new_cost = nearest.cost + np.linalg.norm(new_pos - nearest.pos)
        near_indices = [i for i, n in enumerate(nodes)
                        if np.linalg.norm(n.pos - new_pos) <= rewire_radius]

        best_parent, best_cost = nearest_idx, new_cost
        for ni in near_indices:
            c = nodes[ni].cost + np.linalg.norm(nodes[ni].pos - new_pos)
            if c < best_cost and collision_free(nodes[ni].pos, new_pos, check_obs):
                best_parent, best_cost = ni, c

        new_idx = len(nodes)
        nodes.append(RRTNode(pos=new_pos, parent=best_parent, cost=best_cost))

        for ni in near_indices:
            rc = best_cost + np.linalg.norm(new_pos - nodes[ni].pos)
            if rc < nodes[ni].cost and collision_free(new_pos, nodes[ni].pos, check_obs):
                nodes[ni].parent = new_idx
                nodes[ni].cost = rc

        d2g = np.linalg.norm(new_pos - goal)
        if d2g < goal_tol and collision_free(new_pos, goal, check_obs):
            tc = best_cost + d2g
            if tc < best_goal_cost:
                if goal_idx is None:
                    goal_idx = len(nodes)
                    nodes.append(RRTNode(pos=goal.copy(), parent=new_idx, cost=tc))
                else:
                    nodes[goal_idx].parent = new_idx
                    nodes[goal_idx].cost = tc
                best_goal_cost = tc

    if goal_idx is None:
        return None

    path = []
    idx = goal_idx
    while idx is not None:
        path.append(nodes[idx].pos)
        idx = nodes[idx].parent
    path.reverse()
    return np.array(path)