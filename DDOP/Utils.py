import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from .CorridorGeneration.Corridor_Utils import chebyshev_center, HPolyhedron

def V2H(v):
    hull = ConvexHull(np.array(v))
    A = hull.equations[:,:-1]
    b = -hull.equations[:, -1]
    return A, b

def H2V(A, b):
    """H-rep → vertices"""
    interior = chebyshev_center(HPolyhedron(A,b))
    if interior is None:
        return None
    halfspace = np.column_stack([A,-b])
    try:
        hs = HalfspaceIntersection(halfspace, interior)
        hull = ConvexHull(hs.intersections)
        return hs.intersections[hull.vertices]
    except:pass
    return None
    