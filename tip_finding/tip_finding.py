from cloudvolume import CloudVolume
from meshparty import skeletonize, trimesh_io, meshwork
from caveclient import CAVEclient
import trimesh
import numpy as np


def tip_finder_decimation(root_id, nucleus_id=None, time=None, pt_position=None, sqs_queue_name="None", save_df=False, save_nvq=False):
    datastack_name = "minnie65_phase3_v1"
    client = CAVEclient(datastack_name)
    vol = CloudVolume(
        client.info.segmentation_source(),
        use_https=True,
        progress=False,
        bounded=False,
        fill_missing=True,
        secrets={"token": client.auth.token}
    )
    mesh = vol.mesh.get(str(root_id))
    mesh = mesh[int(root_id)]
    n_faces = mesh.faces.shape[0]
    mesh_obj = trimesh.Trimesh(mesh.vertices, mesh.faces, mesh.normals)
    # test with humphrey
    mesh_obj = trimesh.smoothing.filter_humphrey(mesh_obj, iterations=50)

    # test with laplacian
    # mesh_obj = trimesh.smoothing.filter_laplacian(mesh_obj, iterations=50)
    decimated = trimesh.Trimesh.simplify_quadratic_decimation(
        mesh_obj, n_faces*.50)

    edges_by_component = trimesh.graph.connected_component_labels(
        decimated.face_adjacency)
    largest_component = np.bincount(edges_by_component).argmax()
    largest_component_size = np.sum(edges_by_component == largest_component)

    cc = trimesh.graph.connected_components(
        decimated.face_adjacency, min_len=largest_component_size-1)

    # cc = trimesh.graph.connected_components(
    #     decimated.face_adjacency)

    mask = np.zeros(len(decimated.faces), dtype=bool)

    mask[np.concatenate(cc)] = True

    decimated.update_faces(mask)
    skel = skeletonize.skeletonize_mesh(trimesh_io.Mesh(
        decimated.vertices, decimated.faces), invalidation_d=5000)

    degree_dict = {}
    unique_ids = np.unique(skel.edges)
    for i in unique_ids:
        degree_dict[i] = np.sum(skel.edges == i)

    degree = np.array(list(degree_dict.values()))
    points = skel.vertices[np.argwhere(degree == 1)].astype(int)
    tl = [tuple([int(e[0][0])//4, int(e[0][1])//4, int(e[0][2])//40])
          for e in points]

    # import meshparty
    # import boto3
    # # Upload the file
    # s3_client = boto3.client('s3')
    # try:
    #     # Outputting skeleton to h5 skeleton file for Hannah
    #     # H5 file is then sent to s3 bucket
    #     meshparty.skeleton_io.write_skeleton_h5(skel,f"/root/campfire/data/{nucleus_id}_{root_id}_{time}_skel.h5")
    #     response = s3_client.upload_file(f"/root/campfire/data/{nucleus_id}_{root_id}_{time}_{pt_position}_skel.h5", 'neuvue-skeletons', f"{nucleus_id}_{root_id}_{time}_skel.h5")
    # except Exception as e:
    #     print(e, "s3 upload failed")
    #     save_skel = 'h5'

    return tl, skel, mesh_obj
