from cloudvolume import CloudVolume
from meshparty import skeletonize, trimesh_io, meshwork
from caveclient import CAVEclient
import trimesh
import numpy as np

import datetime
import networkx as nx
from trimesh.voxel import creation as make_voxels


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

def get_mesh(root_id, remove_orphans=True):
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
    if remove_orphans:
        edges_by_component=trimesh.graph.connected_component_labels(mesh_obj.face_adjacency)
        largest_component = np.bincount(edges_by_component).argmax()
        largest_component_size = np.sum(edges_by_component==largest_component)

        cc = trimesh.graph.connected_components(mesh_obj.face_adjacency, min_len=largest_component_size-1)

        mask = np.full(len(mesh_obj.faces), False)

        mask[np.concatenate(cc)] = True

        mesh_obj.update_faces(mask)

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
    # Usually you don't have to use this since trimesh computes face areas for you trivially
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

def normals_next_exp_con(normals, face_inds, face_adjacency, n_faces, area_thresh, face_area, t=.95, mark=1, ct_lim=100):
    
    ids = set()
    ids.update(face_inds)
    o_len = len(ids)
    last_len = 0

    while len(ids) - last_len != 0:
        last_len = len(ids)
        face_adjacency_ids = get_next(ids, face_adjacency, n_faces)
        normals_n = normals[face_adjacency_ids]
        face_adjacency_ids_masked = face_adjacency_ids[np.abs(normals_n[:,2]) >= t]
        new_set = set(face_adjacency_ids_masked)
        new_ids = new_set.difference(ids)
        ids.update(new_set)
        
    expanded_ids = set()
    expanded_ids.update(ids)
    expanded_sum = np.sum(face_area[list(expanded_ids)])
    area_flag = False
    ct = 0
    while not area_flag  and ct < ct_lim:
        if ct == ct_lim:
            print("timeout!")
        face_adjacency_ids = get_next(ids, face_adjacency, n_faces)
        ids.update(face_adjacency_ids)
        new_sum = np.sum(face_area[list(ids)])
        ct+=1
        if new_sum / expanded_sum >= area_thresh:
            area_flag = True

    new_ids = ids.difference(expanded_ids)
    
    return list(ids), list(new_ids)

def get_graphs(mesh_obj, t  = .99, a_min=250, normals_or_z=False):

    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    normals = mesh_obj.face_normals

    if normals_or_z:
        one_pts = np.argwhere(np.abs(normals[:,2]) >= t)
        one_mask = np.full(mesh_obj.faces.shape[0], False)
        one_mask[one_pts] = True
    else:
        diff = mesh_coords[:,:,2] - mesh_coords[:, 0, 2][:, None]
        one_pts = np.argwhere(np.sum(np.abs(diff),axis=1) < t)
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

def get_angles(graphs, mesh_obj, s_min=200000, area_times=2.5, t=.99, area_expand=True):
    sums = np.array([np.sum(list(nx.get_node_attributes(g, 'area').values())) for g in graphs])
    sums_mask =np.squeeze(np.argwhere(sums > s_min))
    print(sums_mask.shape[0], len(graphs), sums.shape)
    if area_expand:
        normals = mesh_obj.face_normals
        means = []
        sums_list = []
        for i, s in enumerate(sums_mask):
            inds = graphs[s].nodes()
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
        graphs_mask = sums_mask[np.array(means_good)]
    else:
        means=[]
        graphs_mask = sums_mask
        sums_list = []

    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    end_points = np.zeros((graphs_mask.shape[0], 3))
    all_points = []
    for i, s in tqdm(enumerate(graphs_mask)):
        node_list = (graphs[s].nodes())

        coords_connected = mesh_coords[node_list]
        coords_expand = coords_connected.reshape(coords_connected.shape[0]*3, 3)
        end_points[i] = np.mean(coords_expand, axis=0)
        all_points.append(coords_expand)

    return means, np.array(sums_list), sums_mask, graphs_mask, end_points, all_points

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

    return skel, np.squeeze(points)

from tqdm import tqdm

def viewer(mesh=None, gt=None, graphs=None, graphs_mask=None, other_points = [], gt_res=np.array([4,4,40])):
    from meshparty import trimesh_vtk
    vis_list = []
    if not mesh == None:
        mesh_actor = trimesh_vtk.mesh_actor(mesh,
                                            color=(1, 1, 0),
                                            opacity=0.5)
        vis_list.append(mesh_actor)

    if not graphs == None:
        mesh_coords = mesh.vertices[mesh.faces]
        for s in tqdm(graphs_mask):
            node_list = (graphs[s].nodes())

            coords_connected = mesh_coords[node_list]
            coords_expand = coords_connected.reshape(coords_connected.shape[0]*3, 3)

            point_actor = trimesh_vtk.point_cloud_actor(coords_expand, size=75, 
                                                        color=(np.random.random(), np.random.random(), np.random.random()),
                                                        opacity=1)#sums[s[0]]/np.max(sums))

            vis_list.append(point_actor)
    
    if not type(gt)==type(None):
        gt_point_actor = trimesh_vtk.point_cloud_actor(gt*gt_res, size=1200, color=(1, 0, 0),
                                                        opacity=.3)
        vis_list.append(gt_point_actor)

    for pc in other_points:
        point_actor = trimesh_vtk.point_cloud_actor(pc, size=75, color=(np.random.random(), np.random.random(), np.random.random()),
                                                        opacity=1)
        vis_list.append(point_actor)
    print("render")
    trimesh_vtk.render_actors(vis_list)

def match_points(mesh, t=.99, a_min=1000, s_min=200000, area_times=3, vox_size=600, invalidation_d=6000, d_thresh=2000, area_expand=True):

    g = get_graphs(mesh, t =t, a_min=a_min)

    ang, sums, sums_mask, sums_mask_good, end_points, all_points = get_angles(g, mesh, s_min=s_min, area_times=area_times, t=t, area_expand=area_expand)   
    
    skel, skel_ep = voxel_skel(mesh, vox_size, soma_radius=100000, invalidation_d=invalidation_d, smooth_neighborhood=2, 
               smooth_iterations=5)

    pt_mask = np.full(end_points.shape[0], True)
    filtered_points = []
    all_points_filtered = [] 
    for i, pt in enumerate(end_points):
        if pt_mask[i]:
            diff = np.sqrt(np.sum(np.square(end_points - pt),axis=1)) < d_thresh
            pt_mask = pt_mask * ~diff
            filtered_points.append(pt)
            all_points_filtered.append(all_points[i])
    filtered_points = np.array(filtered_points)

    true_points = []
    # for i in range(len(filtered_points)):
    # minimum_val=np.inf
    for j in range(filtered_points.shape[0]):
        diff = np.min(np.sqrt(np.sum(np.square(skel_ep - filtered_points[j]), axis=1)))
        if diff < d_thresh:
            # minimum_val = diff
            true_points.append(filtered_points[j])

    # if minimum_val < d_thresh:
    #     true_points.append(filtered_points[i])
            
    true_points = np.array(true_points)


    return true_points, skel_ep, end_points, all_points_filtered

def fit_ellipse(array):
    from skimage.measure import EllipseModel
    # Array is [N,2]
    
    ell = EllipseModel()
    a = ell.estimate(array)
    print(a, array[:4])
    print(ell.params)
    res = ell.residuals(array)
    return np.mean(res)

def find_min_val(point, point_array):
    min_val = np.inf
    for j in range(point_array.shape[0]):
        diff = np.sqrt(np.sum(np.square(point - point_array[j]), axis=1))
        if np.min(diff) < min_val:
            min_val = np.min(diff)
    return min_val

def get_voxel_and_spline(mesh_obj, center, radius, side_len=10):
    from scipy.interpolate import UnivariateSpline
    from scipy.ndimage import binary_fill_holes
    vox = make_voxels.local_voxelize(mesh_obj, center, side_len, radius)
    m=vox.matrix

    m = binary_fill_holes(m)

    coords = np.argwhere(m)
    coords_diff = coords - np.array([radius, radius, radius])
    np.mean(coords_diff, axis=0), np.std(coords_diff,axis=0)
    center_diff = np.mean(coords_diff,axis=0)
    spacing=np.linspace(0,10,100)

    u = np.arange(0,coords.shape[0],100)
    u_small = u[::10]

    # UnivariateSpline
    s = len(u)*1000    # smoothing factor
    spx = UnivariateSpline(u, coords[::100,0], s=s)
    spy = UnivariateSpline(u, coords[::100,1], s=s)
    spz = UnivariateSpline(u, coords[::100,2], s=s)
    #
    xnew = spx(u_small)-radius
    ynew = spy(u_small)-radius
    znew = spz(u_small)-radius

    coords_new = np.vstack([xnew, ynew, znew]).T
    coords_diff = coords_new - radius

    cp = np.argmin(np.sqrt(np.sum(np.square(coords_diff),axis=1)))
    derivative = spx(u_small[cp], 1), spy(u_small[cp], 1), spz(u_small[cp], 1)
    
    center_diff = center_diff / np.linalg.norm(center_diff)
    derivative = derivative / np.linalg.norm(derivative)
    cdiff_min = coords_diff[cp] / np.linalg.norm(coords_diff[cp])

    return derivative, center_diff, cdiff_min

def find_local_diffs(graphs, normals, mesh_obj, sums_mask, radius=50, endpoint=None, gt_dist=200):

    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    direcs = np.zeros(sums_mask.shape[0])

    if type(endpoint) != type(None):
        gt_vec= np.zeros(sums_mask.shape[0])

    dps_diff = np.zeros(sums_mask.shape[0])
    dps = np.zeros(sums_mask.shape[0])
    vecs_diff = np.zeros((sums_mask.shape[0], 3))
    vecs_nearest = np.zeros((sums_mask.shape[0], 3))
    derivs = np.zeros((sums_mask.shape[0], 3))

    for i in tqdm(range(sums_mask.shape[0])):

        face_inds = graphs[sums_mask[i]].nodes()
        direc = np.mean(normals[face_inds][:, 2])
        direcs[i] = np.sign(direc)

        coords = mesh_coords[face_inds]
        coords = coords.reshape(coords.shape[0]*3,3)
        point_center = np.mean(coords, axis=0)

        if type(endpoint) != type(None):
            if find_min_val(endpoint[i], coords) < gt_dist:
                gt_vec[i] = 1
                continue
    

        derivative, center_diff, c_diff_min = get_voxel_and_spline(mesh_obj, point_center, radius, side_len=10)

        vecs_diff[i] = center_diff
        vecs_nearest[i] =c_diff_min

        derivs[i] = derivative / np.linalg.norm(derivative)
        dps_diff[i] = np.dot(center_diff, derivative)

        dps[i] = np.dot(-c_diff_min, derivative )
    return dps, dps_diff, vecs_diff, vecs_nearest, direcs

def get_flat_regions(mesh, minsize=100000, t=1, n_iter=6):
    mesh_obj = trimesh.Trimesh(np.divide(mesh.vertices, np.array([1,1,1])), mesh.faces)
    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    normals = mesh_obj.face_normals

    diff = mesh_coords[:,:,2] - mesh_coords[:, 0, 2][:, None]

    one_pts = np.argwhere(np.sum(np.abs(diff),axis=1) <= 1)
    one_mask = np.full(mesh_obj.faces.shape[0], False)
    one_mask[one_pts] = True

    adjacency_mask =  one_mask[mesh_obj.face_adjacency]
    adjacency_mask_both = adjacency_mask[:,0]*adjacency_mask[:,1]
    E = mesh_obj.face_adjacency[adjacency_mask_both]

    face_areas = mesh_obj.area_faces
    area_dict = {int(i): face_areas[i] for i in range(face_areas.shape[0])}
    face_centers = np.mean(mesh_coords, axis=1)

    loc_dict = {int(i): face_centers[i] for i in range(face_centers.shape[0])}

    G = nx.from_edgelist(E)
    nx.set_node_attributes(G, area_dict, name='area')
    nx.set_node_attributes(G, loc_dict, name='mean_loc')

    degs = G.degree()
    # to_remove = [n for n, d in degs if d != 3 and G.nodes()[n]['area'] < a_min]
    # G.remove_nodes_from(to_remove)

    cc = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]

    graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    sums = np.array([np.sum(list(nx.get_node_attributes(g, 'area').values())) for g in graphs])
    locs = np.array([np.array(list(nx.get_node_attributes(g, 'mean_loc').values())) for g in graphs])
    # face_angles = trimesh.geometry.vector_angle(normals[mesh_obj.face_adjacency])
    # s_min = 100000
    # s_max = 2000000
    sums_mask = np.squeeze(np.argwhere(sums > minsize))
    # sums_mask = np.squeeze(np.argwhere((sums > s_min)))
    # sums_mask.shape[0], len(graphs), sums.shape
    flat_mask = np.full(sums_mask.shape[0], True)
    n_aboves = np.zeros(sums_mask.shape[0])
    n_belows = np.zeros(sums_mask.shape[0])

    for s in tqdm(range(sums_mask.shape[0])):
        curr_nodes = graphs[sums_mask[s]].nodes()
        next_tris = np.zeros(0).astype(int)
        for i in range(n_iter): 
            new_tris = get_next(curr_nodes, mesh_obj.face_adjacency, mesh_obj.faces.shape[0])
            next_tris = np.append(next_tris, new_tris)
            curr_nodes = np.append(curr_nodes, new_tris)

        diffs = mesh_coords[next_tris][:, :, 2] - mesh_coords[curr_nodes][0,0,2]
        n_above = np.sum(diffs > 10)
        n_below = np.sum(diffs < -10)
        n_aboves[s] = n_above
        n_belows[s] = n_below
        if n_above > 0 and n_below > 0:
            flat_mask[s] = False
    sums_mask = sums_mask[flat_mask]

    sums_mask = sums_mask[np.argsort(sums[sums_mask])][::-1]
    
    return sums_mask, locs, sums, graphs, normals

def connected_faces(m, connectivity = 6):
    import cc3d

    m_faces = m.copy()
    m_faces[1:m.shape[0]-1, 1:m.shape[1]-1, 1:m.shape[2]-1] = 0
    labels_out = cc3d.connected_components(m_faces, connectivity=connectivity)
    return labels_out

def fill_faces(m):
    import fill_voids

    m[0] = fill_voids.fill(m[0]).astype(int)
    m[-1] = fill_voids.fill(m[-1]).astype(int)

    m[:, 0] = fill_voids.fill(m[:, 0]).astype(int)
    m[:, -1] = fill_voids.fill(m[:, -1]).astype(int)

    m[:, :, 0] = fill_voids.fill(m[:, :, 0]).astype(int)
    m[:, :, -1] = fill_voids.fill(m[:, :, -1]).astype(int)
    m = fill_voids.fill(m).astype(int)
    return m

def get_largest_component(m):
    import cc3d
    labels_out = cc3d.largest_k(m, k=1, connectivity=26, delta=0)
    m = np.multiply(m, labels_out.T)
    return m

def check_lines(m):
    if np.any(m[0].sum(axis=0)) or np.any(m[0].sum(axis=1)):
        return False
    elif np.any(m[-1].sum(axis=0) > m.shape[0]*.75) or np.any(m[-1].sum(axis=1) > m.shape[0]*.75):
        return False
    elif np.any(m[:, 0].sum(axis=0) > m.shape[0]*.75) or np.any(m[:, 0].sum(axis=1) > m.shape[0]*.75):
        return False
    elif np.any(m[:, -1].sum(axis=0) > m.shape[0]*.75) or np.any(m[:, -1].sum(axis=1) > m.shape[0]*.75):
        return False
    elif np.any(m[:, :, 0].sum(axis=0) > m.shape[0]*.75) or np.any(m[:, :, 0].sum(axis=1) > m.shape[0]*.75):
        return False
    elif np.any(m[:, :, -1].sum(axis=0) > m.shape[0]*.75) or np.any(m[:, :, -1].sum(axis=1) > m.shape[0]*.75):
        return False
    else:
        return True

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
def homogenous_transform(pts, transform):
    pts_homogenous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    tips_nm = np.matmul(transform, pts_homogenous.T)
    return np.round(tips_nm).astype(int).T[:, :3]

def get_mesh_voxel_matrix(mesh, pad=True):
    vox = mesh.voxelized(750)
    forward_transform = vox.transform.copy()
    trans_inv = vox.transform.copy()
    trans_inv[0, 0] = 1 / trans_inv[0, 0]
    trans_inv[1, 1] = 1 / trans_inv[1, 1]
    trans_inv[2, 2] = 1 / trans_inv[2, 2]
    trans_inv[:3, 3] = -1 * (trans_inv[:3, 3] * np.array([trans_inv[0, 0], trans_inv[1, 1], trans_inv[2, 2]]))
    new_mesh = vox.as_boxes()
    m = vox.matrix
    if pad:
        m = np.pad(m, 5)
    # vox = trimesh.voxel.VoxelGrid(trimesh.voxel.morphology.fill(vox.encoding, method='orthographic'), forward_transform)
    m=fill_faces(m)
    return new_mesh, m, forward_transform

def skeletonize_matrix(m, scale=1.5, const=2, anisotropy=1):
    import kimimaro
    DEFAULT_TEASAR_PARAMS = {
        "scale": scale,
        "const": const,
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_acceptance_threshold": 3500,
        "soma_detection_threshold": 750,
        "soma_invalidation_const": 10,
        "soma_invalidation_scale": 1.5
    }

    skels = kimimaro.skeletonize(
    m, 
    teasar_params=DEFAULT_TEASAR_PARAMS,
    # object_ids=[ ... ], # process only the specified labels
    # extra_targets_before=[ (27,33,100), (44,45,46) ], # target points in voxels
    # extra_targets_after=[ (27,33,100), (44,45,46) ], # target points in voxels
    dust_threshold=10, # skip connected components with fewer than this many voxels
    anisotropy=(1,1,anisotropy), # default True
    fix_branching=True, # default True
    fix_borders=True, # default True
    fill_holes=False, # default False
    fix_avocados=False, # default False
    progress=False
    )
    return skels

def get_next_node(edges, pre_node, prev_node = 0):
    ed_with_branch = edges == pre_node
    spots = np.argwhere(ed_with_branch)
    spots[:, 1] = np.abs(spots[:, 1] - 1)
    other_nodes = edges[spots[:,0], spots[:, 1]]
    other_nodes = other_nodes[other_nodes != pre_node]
    other_nodes = other_nodes[other_nodes != prev_node]
    return other_nodes

def remove_small_leaf_branches(branch_nodes, edges, vertices, radii, n_pass=1, len_thresh=10, rad_thresh=1, area_filt=False):
    rad_dict = {}
    len_dict = {}

    mask_verts = np.full(vertices.shape[0], True)

    for _ in range(n_pass):
        rad_dict = {}
        len_dict = {}
        for branch_node in tqdm(branch_nodes):
            next_branch=False
            if next_branch:
                next_branch=False
                continue
            for node in get_next_node(edges, branch_node):
                node = [node]
                prev_node = branch_node
                running_len = []
                radii_list = []
                running_nodes = [node[0]]
                while True:
                    curr_node = node
                    last_pos = vertices[curr_node[0]]
                    radii_list.append(radii[curr_node[0]])
                    node = get_next_node(edges, curr_node, prev_node)

                    prev_node = curr_node
                    new_pos = vertices[node]
                    running_len.append(np.linalg.norm(new_pos-last_pos))
                    if len(node) > 1:
                        break
                    elif len(node) == 0:

                        mean_rad = np.max(radii_list)
                        cable_len = np.sum(running_len)
                        rad_dict[curr_node[0]] = mean_rad
                        len_dict[curr_node[0]] = cable_len
                        if mean_rad <= rad_thresh and cable_len < len_thresh:
                            mask_rows = np.full(edges.shape[0], True)
                            for n in running_nodes:
                                edges_rem = np.argwhere(edges == n)[:, 0]
                                mask_rows[edges_rem] = False
                                mask_verts[n] = False
                            edges = edges[mask_rows]
                        next_branch=True
                        break
                    else:
                        running_nodes.append(node[0])

    area_dict = {key: len_dict[key]*rad_dict[key] for key in len_dict.keys()}
    return edges, area_dict, rad_dict, len_dict, mask_verts

def endpoints_from_rid(root_id, center_collapse=True):
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
    def get_soma(root_id:str):
        cave_client = CAVEclient('minnie65_phase3_v1')
        soma = cave_client.materialize.query_table(
            "nucleus_neuron_svm",
            filter_equal_dict={'pt_root_id':root_id}
        )
        return soma

    s = get_soma(str(root_id))
    if s.shape[0] > 0:
        soma_center = s.pt_position.iloc[0] * np.array([4,4,40])
    else:
        soma_center=None
        print(root_id, "No Soma Found")
    tips_thick, tips_thin, thru_branch_tips, tip_no_flat_thick, tip_no_flat_thin, flat_no_tip = get_endpoints(mesh, soma_center)

    return tips_thick, tips_thin, thru_branch_tips, tip_no_flat_thick, tip_no_flat_thin, flat_no_tip 


def get_endpoints(mesh, center=None, invalidation=3000, len_thresh=6000, rad_thresh=6000, remove_small=True, path_dist_to_tip=5000):
    if type(center) == type(None):
        collapse_function = 'branch'
    else:
        collapse_function = 'sphere'
    skel_mp = skeletonize.skeletonize_mesh(trimesh_io.Mesh(mesh.vertices, 
                                                mesh.faces),
                                                invalidation_d=invalidation,
                                                collapse_function=collapse_function,
                                                soma_radius = 7500,
                                                soma_pt=center,
                                                smooth_neighborhood=5,
#                                                     collapse_params = {'dynamic_threshold':True}
                                                )
    sums_mask, means, sums, graphs, normals = get_flat_regions(mesh)

    flat_edges = skel_mp.edges.flatten()
    edge_bins = np.bincount(flat_edges)
    eps = np.squeeze(np.argwhere(edge_bins==1))
    bps = np.squeeze(np.argwhere(edge_bins==3))

    endpoints_nm = skel_mp.vertices[eps]

    flat_tip_agree = {}
    flat_tip_branch = []
    extra_flat_surfaces_tip = []
    flat_no_tip = []

    means_mask = means[sums_mask]
    hit_tips = np.zeros(endpoints_nm.shape[0])+path_dist_to_tip
    for i, tip in tqdm(enumerate(means_mask)):
        tip = np.mean(tip, axis=0)
    #     for j, pt in enumerate(tip):
        closest_skel_pt_dist = np.linalg.norm(skel_mp.vertices - tip, axis=1)
        if np.min(closest_skel_pt_dist) < 2000:
            closest_pt = np.argmin(closest_skel_pt_dist)

            tip_dist = np.linalg.norm(endpoints_nm - tip, axis=1)
            tip_hit = np.argmin(tip_dist)
            closest_tip = eps[tip_hit]
            node_to_tip_path = skel_mp.path_between(closest_pt, closest_tip)
            min_tip = np.min(tip_dist)

            if min_tip < hit_tips[tip_hit] and np.any(np.isin(node_to_tip_path, bps)):
                flat_tip_branch.append(tip)
                print("branch path")
                continue
            if min_tip < hit_tips[tip_hit]:
                if tip_hit in flat_tip_agree:
                    extra_flat_surfaces_tip.append(flat_tip_agree[tip_hit])
                flat_tip_agree[tip_hit] = tip
                hit_tips[tip_hit] = min_tip
            elif tip_hit in flat_tip_agree:
                extra_flat_surfaces_tip.append(tip)
            else:
                flat_no_tip.append(tip)
    if remove_small:
        edges_thick, area_dict, rad_dict, len_dict, mask_verts = remove_small_leaf_branches(skel_mp.branch_points, 
                                                            skel_mp.edges,
                                                            skel_mp.vertices,
                                                            skel_mp.radius, 
                                                            len_thresh=len_thresh, 
                                                            rad_thresh=rad_thresh)
        flat_tip_agree_thin = {}
        flat_tip_agree_thick = {}
        edges_thick_flat  = edges_thick.flatten()
        edge_bins_thick = np.bincount(edges_thick_flat)
        eps_thick = np.squeeze(np.argwhere(edge_bins_thick==1))
        for f in flat_tip_agree:
            if f in eps_thick:
                flat_tip_agree_thick[f] = flat_tip_agree[f]
            else:
                flat_tip_agree_thin[f] = flat_tip_agree[f]
        thick_mask = np.isin(eps, eps_thick)
    else:
        thick_mask = np.ones(eps.shape[0])
        flat_tip_agree_thin = {}
        flat_tip_agree_thick = flat_tip_agree
    tip_no_flat_thick = eps[(hit_tips == path_dist_to_tip) * thick_mask]
    tip_no_flat_thin = eps[(hit_tips == path_dist_to_tip) * ~thick_mask]
    
    flat_tip_branch = np.array(flat_tip_branch)
    # flat_tip_agree = np.array(list(flat_tip_agree.values()))
    flat_tip_agree_thick = np.array(list(flat_tip_agree_thick.values()))

    flat_tip_agree_thin = np.array(list(flat_tip_agree_thin.values()))
    flat_no_tip = np.array(flat_no_tip)
    
    return flat_tip_agree_thick, flat_tip_agree_thin, flat_tip_branch, tip_no_flat_thick, tip_no_flat_thin, flat_no_tip