import numpy as np
from sklearn import neighbors

import scipy

import trimesh


def to_unit_vector(vector):
    """
    Unitizes given vector. Also capable of handling matricies, where rows are isolated.
    """
    vec_arr = np.asanyarray(vector)

    # Set threshold to remove floating-point inaccuracy
    thresh = np.finfo(np.float64).resolution
    thresh *= 100

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
    
    return unitized_vector
    

def neighbor_vertex_norms(mesh):
    """
    Helper function to get vertex normals. Bias from neighboring faces not considered.
    """
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        face_norms = mesh.face_normals
    except:
        raise AttributeError('Mesh object lacks necessary attributes.')

    vertex_norms = trimesh.geometry.index_sparse(len(vertices), faces).dot(face_norms)
    neighbor_norms = to_unit_vector(vertex_norms)

    return neighbor_norms


def derivative_dilate(vertices, faces, norms, vol, finite_diffs):
    """
    Helper function to get dilation derivative of sets of vertices.
    """
    verts_to_dilate = vertices + norms*finite_diffs

    vertices_dilated = trimesh.triangles.mass_properties(verts_to_dilate[faces], skip_inertia=True)['volume']

    ddx_vertices_dilated = (finite_diffs) / (vertices_dilated - vol)

    return ddx_vertices_dilated


def base_laplacian(mesh, use_inverse=False, static_nodes=[]):
    """
     Helper function, gets sparse matrix (as defined by SciPy standard) for
     downstream Laplacian methods.
    """
    try:
        vertex_neighbors = mesh.vertex_neighbors
        mesh.vertices
    except:
        raise AttributeError('Mesh object lacks necessary attributes')

    # Cannot use asanyarray as it will pass the TrackedArray through
    vertices = np.asarray(mesh.vertices)

    for node in static_nodes:
        vertex_neighbors[node] = [node]

    n = np.concatenate(vertex_neighbors)
    m = np.concatenate([idx]*len(i) for idx, i in enumerate(neighbors))

    if not use_inverse:
        matrix = np.concatenate([1.0/len(n)]*len(n) for n in vertex_neighbors)
    else:
        nm = [1.0/np.maximum(1e-6, np.sqrt(np.dot(
            vertices[idx] - vertices[i]) ** 2, np.ones(3))) for idx, i in enumerate(vertex_neighbors)]
        
        matrix = np.concatenate([i / i.sum() for i in nm])
    
    matrix = scipy.sparse.coo_matrix((matrix, (m, n)), shape=[len(vertices)]*2)

    return matrix


if __name__ == "__main__":
    import sys
    sys.path.append('$HOME/campfire/test_modules/')
    from test_modules.primary_wrapper import wrapper_run
    wrapper_run(864691136184827094, [to_unit_vector, derivative_dilate, base_laplacian], in_order=True)
