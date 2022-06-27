from cloudvolume import CloudVolume
from meshparty import skeletonize, trimesh_io, meshwork
from caveclient import CAVEclient
import trimesh
import numpy as np
import meshparty
import boto3
import datetime
import networkx as nx

def get_skel(root_id):
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

    if n_faces < 1000:
        return 'stub'
    skel = skeletonize.skeletonize_mesh(trimesh_io.Mesh(mesh.vertices, 
                                                                mesh.faces))
    return skel

def get_mesh(root_id):
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
    mesh_obj = trimesh.Trimesh(mesh.vertices, mesh.faces, mesh.normals)
    return mesh_obj

def tip_finder_decimation(root_id, decimation = .05, nucleus_id=None, time=None, pt_position=None, sqs_queue_name = "None", save_df=False, save_nvq=False,
                            soma_thresh=5000,
                            invalidation_d=10000,
                            smooth_neighborhood=5,
                            large_skel_path_threshold=5000):
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
    if n_faces < 1000:
        if not pt_position:
            return [tuple(mesh_obj.vertices.mean(axis=0).astype('int'))]
        else:
            return mesh_obj.vertices[np.argmin(np.sqrt(np.sum(np.square(mesh_obj.vertices - pt_position), axis=1)))]
    
    mesh_obj = trimesh.Trimesh(mesh.vertices, mesh.faces, mesh.normals)
    decimated = trimesh.Trimesh.simplify_quadratic_decimation(mesh_obj, n_faces*decimation)

    edges_by_component=trimesh.graph.connected_component_labels(decimated.face_adjacency)
    largest_component = np.bincount(edges_by_component).argmax()
    largest_component_size = np.sum(edges_by_component==largest_component)

    cc = trimesh.graph.connected_components(decimated.face_adjacency, min_len=largest_component_size-1)

    mask = np.zeros(len(decimated.faces), dtype=bool)

    mask[np.concatenate(cc)] = True

    decimated.update_faces(mask)
    skel = skeletonize.skeletonize_mesh(trimesh_io.Mesh(decimated.vertices, 
                                                        decimated.faces,
                                                        soma_thresh=soma_thresh,
                                                        invalidation_d=invalidation_d,
                                                        smooth_neighborhood=smooth_neighborhood,
                                                        large_skel_path_threshold=large_skel_path_threshold,
                                                        cc_vertex_thresh=largest_component_size-1))

    degree_dict = {}
    unique_ids = np.unique(skel.edges)
    for i in unique_ids:
        degree_dict[i] = np.sum(skel.edges == i)
        
    degree = np.array(list(degree_dict.values()))
    points = skel.vertices[np.argwhere(degree==1)].astype(int)
    tl = [tuple([int(e[0][0])//4, int(e[0][1])//4, int(e[0][2])//40]) for e in points]

    # Upload the file
    # s3_client = boto3.client('s3')
    # try:
    #     # Outputting skeleton to h5 skeleton file for Hannah
    #     # H5 file is then sent to s3 bucket
    #     meshparty.skeleton_io.write_skeleton_h5(skel,f"/root/campfire/data/{nucleus_id}_{root_id}_{time}_skel.h5")
    #     response = s3_client.upload_file(f"/root/campfire/data/{nucleus_id}_{root_id}_{time}_{pt_position}_skel.h5", 'neuvue-skeletons', f"{nucleus_id}_{root_id}_{time}_skel.h5")
    # except Exception as e:
    #     print(e, "s3 upload failed")
    #     save_skel = 'h5'
    
    return tl, skel, decimated

def check_axon(point, seg_id, distance=2500):

    point_a = [coord - distance for coord in point]
    point_b = [coord + distance for coord in point]
    bounding_box = [point_a, point_b]

    client = CAVEclient('minnie65_phase3_v1')
    synapse_table = 'synapses_pni_2'
    pre_ids = client.materialize.query_table(synapse_table,
        filter_equal_dict={'pre_pt_root_id': seg_id},
        filter_spatial_dict={'pre_pt_position': bounding_box},
        select_columns=['pre_pt_root_id','pre_pt_position'],
        timestamp = datetime.datetime(2022, 4, 29)
    )
    post_ids = client.materialize.query_table(synapse_table,
        filter_equal_dict={'post_pt_root_id': seg_id},
        filter_spatial_dict={'post_pt_position': bounding_box},
        select_columns=['post_pt_root_id','post_pt_position'],
        timestamp = datetime.datetime(2022, 4, 29)
    )
    return len(pre_ids), len(post_ids)

## Face AREA ## 
def calc_area(coords):
    coords_xy = coords[:, :, :2]

    coords_xy_a, coords_xy_b, coords_xy_c = coords_xy[:, 0], coords_xy[:, 1], coords_xy[:, 2]

    t1 = np.multiply(coords_xy_a[:, 0], coords_xy_b[:, 1] - coords_xy_c[:, 1])
    t2 = np.multiply(coords_xy_b[:, 0], coords_xy_c[:, 1] - coords_xy_a[:, 1])
    t3 = np.multiply(coords_xy_c[:, 0], coords_xy_a[:, 1] - coords_xy_b[:, 1])

    face_area = .5* np.array(np.abs(t1 + t2 + t3))
    return face_area

def get_next(ids, ad, n):
    mask = np.full(n, False)
    mask[list(ids)] = True

    zero_mask = mask[ad]
    zero_mask_rows = zero_mask[:,0] + zero_mask[:,1]
    fa_sub = ad[zero_mask_rows]
    zero_mask = zero_mask[zero_mask_rows]
    zero_mask = ~zero_mask
    fais = fa_sub[zero_mask]
    
    return fais

def normals_next_exp_con(normals, face_inds, face_adjacency, n_faces, area_thresh, face_area, t=.95, mark=1):
    
    ids = set()
    ids.update(face_inds)
    o_len = len(ids)
    last_len = 0

    while len(ids) - last_len != 0:
        last_len = len(ids)
        face_adjacency_ids = get_next(ids, face_adjacency, n_faces)
        normals_n = normals[face_adjacency_ids]
        face_adjacency_ids_masked = face_adjacency_ids[normals_n[:,2] >= t]
        new_set = set(face_adjacency_ids_masked)
        new_ids = ids.difference(new_set)
        ids.update(new_set)
        
    expanded_ids = set()
    expanded_ids.update(ids)
    expanded_sum = np.sum(face_area[list(expanded_ids)])
    area_flag = False
    ct = 0
    while not area_flag  and ct < 100:
        face_adjacency_ids = get_next(ids, face_adjacency, n_faces)
        ids.update(face_adjacency_ids)
        new_sum = np.sum(face_area[list(ids)])
        ct+=1
        if new_sum / expanded_sum >= area_thresh:
            area_flag = True

    new_ids = ids.difference(expanded_ids)
    
    return list(ids), list(new_ids)

def get_graphs(mesh_obj, t  = .99, a_min=250):

    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    normals = mesh_obj.face_normals

    one_pts = np.argwhere(np.abs(normals[:,2]) >= t)
    one_mask = np.full(mesh_obj.faces.shape[0], False)
    one_mask[one_pts] = True

    adjacency_mask =  one_mask[mesh_obj.face_adjacency]
    adjacency_mask_both = adjacency_mask[:,0]*adjacency_mask[:,1]
    E = mesh_obj.face_adjacency[adjacency_mask_both]

    face_areas = mesh_obj.area_faces
    area_dict = {int(i): face_areas[i] for i in range(face_areas.shape[0])}

    G = nx.from_edgelist(E)
    nx.set_node_attributes(G, area_dict, name='area')

    degs = G.degree()
    to_remove = [n for n, d in degs if d != 3 and G.nodes()[n]['area'] < a_min]
    G.remove_nodes_from(to_remove)

    graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    return graphs

def get_angles(graphs, mesh_obj, s_min=200000, area_times=2.5, t=.99):
    sums = np.array([np.sum(list(nx.get_node_attributes(g, 'area').values())) for g in graphs])
    sums_mask = np.argwhere(sums > s_min)
    print(sums_mask.shape[0], len(graphs), sums.shape)
    normals = mesh_obj.face_normals
    means = []
    sums_list = []
    for i, s in enumerate(sums_mask):
        inds = graphs[s[0]].nodes()
        all_ids, new_ids = normals_next_exp_con(normals, inds, mesh_obj.face_adjacency, 
                                                mesh_obj.faces.shape[0], area_times, mesh_obj.area_faces, t=t)
        all_normals = normals[all_ids]
        new_normals = normals[new_ids]
        point_dir = np.mean(normals[inds][:, 2])

        ang = np.arccos(all_normals[:, 2] * point_dir)
        means.append(np.mean(ang))
        sums_list.append(np.sum(ang < 1))
    means=np.array(means)
    means_good = means > 1.05
    return means, np.array(sums_list), sums_mask, sums_mask[np.array(means_good)]


def voxel_skel(mesh, vox_n=300, soma_radius=10000, invalidation_d=10000, smooth_neighborhood=1, 
               smooth_iterations=5, cc_vertex_thresh=100):
    vox = mesh.voxelized(vox_n)
    new_mesh = vox.as_boxes()

    skel = skeletonize.skeletonize_mesh(trimesh_io.Mesh(new_mesh.vertices, 
                                                        new_mesh.faces),
                                                        soma_radius=soma_radius,
                                                        invalidation_d=invalidation_d,
                                                        smooth_neighborhood=smooth_neighborhood,
                                                        smooth_iterations=smooth_iterations,
                                                        cc_vertex_thresh=cc_vertex_thresh)
    degree_dict = {}
    unique_ids = np.unique(skel.edges)
    for i in unique_ids:
        degree_dict[i] = np.sum(skel.edges == i)

    degree = np.array(list(degree_dict.values()))
    points = skel.vertices[np.argwhere(degree==1)].astype(int)

    return skel, points

from meshparty import trimesh_vtk
from tqdm import tqdm

def viewer(mesh=None, gt=None, graphs=None, graphs_mask=None, gt_res=np.array([4,4,40])):
    vis_list = []
    if not mesh == None:
        mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                            color=(1, 1, 0),
                                            opacity=0.5)
        vis_list.append(mesh_actor)

    if not graphs == None:
        mesh_coords = mesh.vertices[mesh.faces]
        for s in tqdm(graphs_mask):
            node_list = (graphs[s[0]].nodes())
            mask = np.full(mesh.faces.shape[0], False)
            mask[node_list] = True
            coords_connected = mesh_coords[mask]
            coords_expand = coords_connected.reshape(coords_connected.shape[0]*3, 3)

            point_actor = trimesh_vtk.point_cloud_actor(coords_expand, size=300, 
                                                        color=(np.random.random(), np.random.random(), np.random.random()),
                                                        opacity=1)#sums[s[0]]/np.max(sums))

            vis_list.append(point_actor)
    if not type(gt)==type(None):
        gt_point_actor = trimesh_vtk.point_cloud_actor(gt*gt_res, size=600, color=(1, 0, 0),
                                                        opacity=.5)
        vis_list.append(gt_point_actor)
    print("render")
    trimesh_vtk.render_actors(vis_list)