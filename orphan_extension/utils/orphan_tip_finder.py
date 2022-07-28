from cloudvolume import CloudVolume
from meshparty import skeletonize, trimesh_io, meshwork
from caveclient import CAVEclient
import trimesh
import numpy as np

from orphan_mesh import Smoothing


def orphan_tip_finder(
    root_id,
    decimation_level=0.5,
    nucleus_id=None,
    time=None,
    pt_position=None,
    sqs_queue_name="None",
    save_df=False,
    save_nvq=False,
    inval_d=5000,
    num_smoothing_iters=20,
    decimation_factor=0.5,
    apply_smoothing=True,
):
    smoothing_engine = Smoothing(root_id)
    n_faces = smoothing_engine.faces.shape[0]
    if apply_smoothing:
        mesh_obj = smoothing_engine.bilaplacian_smoothing(
            smoothing_passes=num_smoothing_iters, smooth=decimation_factor
        )
    else:
        mesh_obj = smoothing_engine.mesh

    decimated = trimesh.Trimesh.simplify_quadratic_decimation(
        mesh_obj, n_faces * decimation_level
    )

    edges_by_component = trimesh.graph.connected_component_labels(
        decimated.face_adjacency
    )
    largest_component = np.bincount(edges_by_component).argmax()
    largest_component_size = np.sum(edges_by_component == largest_component)

    cc = trimesh.graph.connected_components(
        decimated.face_adjacency, min_len=largest_component_size - 1
    )

    # cc2 = trimesh.graph.connected_components(decimated.face_adjacency)

    mask = np.zeros(len(decimated.faces), dtype=bool)

    mask[np.concatenate(cc)] = True

    decimated.update_faces(mask)
    skel = skeletonize.skeletonize_mesh(
        trimesh_io.Mesh(decimated.vertices, decimated.faces), invalidation_d=inval_d
    )

    degree_dict = {}
    unique_ids = np.unique(skel.edges)
    for i in unique_ids:
        degree_dict[i] = np.sum(skel.edges == i)

    degree = np.array(list(degree_dict.values()))
    points = skel.vertices[np.argwhere(degree == 1)].astype(int)
    tl = [
        tuple([int(e[0][0]) // 4, int(e[0][1]) // 4, int(e[0][2]) // 40])
        for e in points
    ]

    return tl, skel, mesh_obj


if __name__ == "__main__":
    t1, skel, mesh_obj = orphan_tip_finder(864691136020754778)
    # WORKING SEG ID: 864691136700953198
    print(skel)
