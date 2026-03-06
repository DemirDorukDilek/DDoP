import numpy as np
from scipy.spatial import ConvexHull

def V2H(vertices):
    vertices = np.array(vertices, dtype=float)
    hull = ConvexHull(vertices)
    vertices = vertices[hull.vertices]
    
    centroid = vertices.mean(axis=0)
    n = len(vertices)
    A = np.empty((n, 2))
    b = np.empty(n)
    
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        normal = np.array([edge[1], -edge[0]])
        
        # normal içe baksın
        if normal @ (centroid - vertices[i]) < 0:
            normal = -normal
        
        A[i] = normal
        b[i] = normal @ vertices[i]
    
    return A, b

def H2V(A, b):
    """H-rep → vertices"""
    n = len(A)
    vertices = []
    
    for i in range(n):
        for j in range(i + 1, n):
            det = A[i, 0] * A[j, 1] - A[i, 1] * A[j, 0]
            if abs(det) < 1e-10:
                continue
            
            x = (b[i] * A[j, 1] - b[j] * A[i, 1]) / det
            y = (A[i, 0] * b[j] - A[j, 0] * b[i]) / det
            vertex = np.array([x, y])
            
            if np.all(A @ vertex <= b + 1e-6):
                vertices.append(vertex)
    
    if len(vertices) < 3:
        return None
    
    vertices = np.unique(np.round(vertices, 6), axis=0)
    if len(vertices) < 3:
        return None
    
    center = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angles)]
    