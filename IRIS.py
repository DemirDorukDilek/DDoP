import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from typing import List, Tuple, Optional
from Ucorridor import *


def _compute_hyperplanes(center, obstacles, bounds):
    dim = len(center)
    A_rows, b_rows = [], []
    for obs in obstacles:
        p = closest_point_on_obstacle(center, obs)
        n = p - center
        if np.linalg.norm(n) < 1e-10:
            continue
        A_rows.append(n)
        b_rows.append(n @ p)
    lower, upper = bounds
    for i in range(dim):
        e = np.zeros(dim); e[i] = 1.0
        A_rows.append(e.copy());  b_rows.append(upper[i])
        A_rows.append(-e.copy()); b_rows.append(-lower[i])
    return HPolyhedron(np.array(A_rows), np.array(b_rows))


def _max_ellipsoid(hpoly):
    A, b = hpoly.A, hpoly.b
    m, dim = A.shape
    C = cp.Variable((dim, dim), symmetric=True)
    d = cp.Variable(dim)
    constraints = [cp.norm(C @ A[i], 2) + A[i] @ d <= b[i] for i in range(m)]
    constraints.append(C >> 0)
    prob = cp.Problem(cp.Maximize(cp.log_det(C)), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
        if prob.status in ['optimal', 'optimal_inaccurate'] and C.value is not None:
            Cv = (C.value + C.value.T) / 2
            eigv = np.linalg.eigvalsh(Cv)
            Cv += max(0, -np.min(eigv) + 1e-8) * np.eye(dim)
            return Ellipsoid(Cv, d.value)
    except Exception:
        pass
    return None


def iris(seed, obstacles, bounds, max_iter=5, tol=1e-2):
    center = seed.copy()
    prev_vol = 0.0
    best_hpoly, best_e = None, None

    for _ in range(max_iter):
        hpoly = _compute_hyperplanes(center, obstacles, bounds)
        if not hpoly.contains(seed):
            break
        e = _max_ellipsoid(hpoly)
        if e is None:
            break
        best_hpoly, best_e = hpoly, e
        center = e.d
        vol = e.volume
        if prev_vol > 0 and abs(vol - prev_vol) / (prev_vol + 1e-15) < tol:
            break
        prev_vol = vol

    return Corridor(best_hpoly, best_e, seed)


def _corridor_coverage(corridor, path, start_idx):
    n = len(path)
    end = start_idx
    for i in range(start_idx, n):
        if corridor.hpoly.contains(path[i]):
            end = i + 1
        else:
            break
    return end


def greedy_corridor_generation(path, obstacles, bounds, max_corridors=30, min_cheb_radius=0.01,obstacle_margin=0.05):
    n = len(path)
    if n < 2:
        return None
    inflated = [inflate_obstacle(o, obstacle_margin) for o in obstacles]

    first = iris(path[0].copy(), inflated, bounds)
    if first is None:
        return None

    corridors = [first]
    waypoints = [path[0].copy()]
    last_best_seed = 0
    radii = []

    covered_until = _corridor_coverage(first, path, 0)

    while covered_until < n and len(corridors) < max_corridors:
        prev = corridors[-1]
        best_candidate = None
        best_seed = None
        best_candidate_coverage = covered_until
        best_cheb_center = None
        best_cheb_radius = 0.0

        scan_idx = covered_until if last_best_seed >= covered_until-1 else covered_until-1
        while scan_idx < n:
            trial = iris(path[scan_idx].copy(), inflated, bounds)
            if trial is None:
                scan_idx += 1
                continue

            h_inter = intersect_hpoly(prev.hpoly, trial.hpoly)
            cheb = chebyshev_center(h_inter)

            if cheb is not None:
                A, b = h_inter.A, h_inter.b
                r = np.min((b - A @ cheb) / (np.linalg.norm(A, axis=1) + 1e-15))

                if r >= min_cheb_radius:
                    trial_coverage = _corridor_coverage(trial, path, scan_idx)
                    best_candidate = trial
                    best_seed = scan_idx
                    best_candidate_coverage = trial_coverage
                    best_cheb_center = cheb
                    best_cheb_radius = r

                    scan_idx = trial_coverage
                    continue

            if best_candidate is not None:
                break

            trial_coverage = _corridor_coverage(trial, path, scan_idx)
            if trial_coverage > scan_idx:
                scan_idx = trial_coverage
            else:
                scan_idx += 1

        if best_candidate is not None:
            corridors.append(best_candidate)
            waypoints.append(best_cheb_center)
            radii.append(best_cheb_radius)
            last_best_seed = best_seed
            covered_until = best_candidate_coverage
        else:
            print(f"FAIL: No intersecting corridor found from idx {covered_until}")
            return None

    waypoints.append(path[-1].copy())

    if corridors[1].hpoly.contains(waypoints[0]):
        corridors.pop(0)
        waypoints.pop(1)
        radii.pop(0)

    return corridors, np.array(waypoints), radii




def verify_corridors(corridors, radii):
    ok = True
    for i in range(len(corridors) - 1):
        if radii[i] <= 0:
            print(f"    WARN: Corridors {i} and {i+1} have zero intersection radius")
            ok = False
    if ok:
        print("    OK")
    return ok