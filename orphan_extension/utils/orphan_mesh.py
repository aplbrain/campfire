import numpy as np

import trimesh
from trimesh.triangles import mass_properties
from trimesh.geometry import index_sparse
from trimesh.util import unitize

    
def to_unit_vector(vector, validate=False, thresh=None, thresh_scalar=100):
    vec_arr = np.asanyarray(vector)

    if not thresh:
        thresh = np.finfo(np.float64).resolution
        thresh *= thresh_scalar

    if len(vec_arr.shape) == 2:
        nm = np.sqrt(np.dot(vec_arr*vec_arr, [1.0]*vec_arr.shape[1]))
        non_zero_norms = nm > thresh

        nm[non_zero_norms] **= -1
        unitized_vector = vec_arr * nm.reshape((-1,1))
    
    elif len(vec_arr.shape) == 1:
        nm = np.sqrt(np.dot(vec_arr, vec_arr))
        non_zero_norms = nm > thresh

        if non_zero_norms:
            unitized_vector = vec_arr / nm
        else:
            unitized_vector = vec_arr.copy()
    
    else:
        raise ArithmeticError('Internal vector dimension error.')
    
    if validate:
        return unitized_vector[non_zero_norms], non_zero_norms

    else:
        return unitized_vector
    

def neighbor_vertex_norms(mesh):
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        face_norms = mesh.face_normals
    except:
        raise AttributeError('Mesh object lacks necessary attributes.')

    vertex_norms = index_sparse(len(vertices), faces).dot(face_norms)
    neighbor_norms = to_unit_vector(vertex_norms)

    return neighbor_norms


def derivative_dilate(vertices, faces, norms, vol, finite_diffs):
    vortices = vertices + norms*finite_diffs

    v = mass_properties(vortices[faces], skip_inertia=True)['volume']

    return (finite_diffs) / (v - vol)