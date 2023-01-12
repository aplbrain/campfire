from cloudvolume import CloudVolume
import backoff
from meshparty import skeletonize, trimesh_io, meshwork
from caveclient import CAVEclient
import trimesh
import numpy as np

import datetime
import networkx as nx
from trimesh.voxel import creation as make_voxels
from scipy.sparse import identity
from scipy.spatial import distance_matrix
import scipy 
from tqdm import tqdm

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
    trimesh_vtk.render_actors(vis_list)

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

def get_soma(soma_id:str):
    cave_client = CAVEclient('minnie65_phase3_v1')
    soma = cave_client.materialize.query_table(
        "nucleus_neuron_svm",
        filter_equal_dict={'id':soma_id}
    )
    return soma

#@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def error_locs_defects(root_id, soma_id = None, soma_table=None, center_collapse=True):
    #print("START", root_id)
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

    mesh = vol.mesh.get(str(root_id))[root_id]
    mesh_obj = trimesh.Trimesh(np.divide(mesh.vertices, np.array([1,1,1])), mesh.faces)

    if mesh_obj.volume > 5000000000000:
        print("TOO BIG, SKIPPING")
    trimesh.repair.fix_normals(mesh_obj)
    mesh_obj.fill_holes()
    ccs_graph = trimesh.graph.connected_components(mesh_obj.edges)
    ccs_len = [len(c) for c in ccs_graph]

    try:
        if soma_table==None:
            soma_table = get_soma(str(soma_id))
        if soma_table[soma_table.id == soma_id].shape[0] > 0:
            center = np.array(soma_table[soma_table.id == soma_id].pt_position)[0] * [4,4,40]
        else:
            center=None
    except:
        center = None

    # Subselect the parts of the mesh that are not inside one another 
    # the other components are an artifact of the soma seg and small unfilled sections
    largest_component = ccs_graph[np.argmax(ccs_len)]
    largest_component_remap = np.arange(ccs_graph[np.argmax(ccs_len)].shape[0])
    face_dict = {largest_component[i]:largest_component_remap[i] for i in range(largest_component.shape[0])}

    new_faces_mask = np.isin(mesh_obj.faces, list(face_dict.keys()))
    new_faces_mask = new_faces_mask[:, 0]*new_faces_mask[:, 1]*new_faces_mask[:, 2]

    new_faces = np.vectorize(face_dict.get)(mesh_obj.faces[new_faces_mask])
    new_faces = new_faces[new_faces[:, 0] != None]
    largest_component_mesh = trimesh.Trimesh(mesh_obj.vertices[largest_component], new_faces)

    all_ids = set(largest_component)
    encapsulated_ids = []

    for i in range(1, len(ccs_graph)):
        n_con = largest_component_mesh.contains(mesh_obj.vertices[ccs_graph[i]])
        if np.sum(n_con) / n_con.shape[0] == 0 and n_con.shape[0] > 50:
            all_ids.update(ccs_graph[i])
        else:
            if len(ccs_graph[i]) < 1000:
                encapsulated_ids.append((np.mean(mesh_obj.vertices[ccs_graph[i]], axis=0)/[4,4,40], len(ccs_graph[i])))
            
    all_component = np.array(list(all_ids))
    all_component_remap = np.arange(all_component.shape[0])
    face_dict = {all_component[i]:all_component_remap[i] for i in range(all_component.shape[0])}
    new_faces_mask = np.isin(mesh_obj.faces, list(face_dict.keys()))
    new_faces_mask = new_faces_mask[:, 0]*new_faces_mask[:, 1]*new_faces_mask[:, 2]

    new_faces = np.vectorize(face_dict.get)(mesh_obj.faces[new_faces_mask])
    new_faces[new_faces[:, 0] != None]
    
    largest_component_mesh = trimesh.Trimesh(mesh_obj.vertices[all_component], new_faces)

    mesh_obj = largest_component_mesh
    # find edges that only occur once..  might be faster to find these in the sparse matrix..
    bad_edges = trimesh.grouping.group_rows(
        mesh_obj.edges_sorted, require_count=1)
    bad_edges_ind = mesh_obj.edges[bad_edges]
    sparse_edges = mesh_obj.edges_sparse
    xs = list(bad_edges_ind[:, 0]) + list(bad_edges_ind[:, 1]) 
    ys = list(bad_edges_ind[:, 1]) + list(bad_edges_ind[:, 0])
    vs = [1]*bad_edges_ind.shape[0]*2
    bad_inds = scipy.sparse.coo_matrix((vs, (xs, ys)), shape=(mesh_obj.vertices.shape[0], mesh_obj.vertices.shape[0]))
    # Make it symmetrical and add identity so each integrates from itself too, then subtract singleton edges
    # I noticed that the number of asymmetrical edges vs the number of single edges I find from group rows
    # Are close but different. Haven't looked into that yet. Also removing edges 1 hop away from single edges to remove bias towards
    # Holes in the mesh that are caused by mesh construction errors as opposed to segmentation errors
    sparse_edges = mesh_obj.edges_sparse + mesh_obj.edges_sparse.T + identity(mesh_obj.edges_sparse.shape[0]) - sparse_edges.multiply(bad_inds) - bad_inds
    degs = mesh_obj.vertex_degree + 1

    # N_iter is a smoothing parameter here. The loop below smooths the vertex error about the mesh to get more consistent connected regions
    n_iter = 2
    angle_sum = np.array(abs(mesh_obj.face_angles_sparse).sum(axis=1)).flatten()
    defs = (2 * np.pi) - angle_sum

    abs_defs = np.abs(defs)
    abs_defs_i = abs_defs.copy()
    for i in range(n_iter):
        abs_defs_i = sparse_edges.dot(abs_defs_i) / degs
    
    a = .75

    verts_select = np.argwhere((abs_defs_i > a))# & (abs_defs < 2.5))
    edges_mask = np.isin(mesh_obj.edges, verts_select)
    edges_mask[bad_edges] = False
    edges_select = edges_mask[:, 0] * edges_mask[:, 1]
    edges_select = mesh_obj.edges[edges_select]

    G = nx.from_edgelist(edges_select)#f_edge_sub)

    ccs = nx.connected_components(G)
    subgraphs = [G.subgraph(cc).copy() for cc in ccs]

    lens = []
    lengths = []
    for i in tqdm(range(len(subgraphs))):
        ns = np.array(list(subgraphs[i].nodes()))
    #     ns = ns[abs_defs[ns ]]
        l = len(ns)
        if l > 20 and l < 5000:
            lens.append(ns)
            lengths.append(l)
    lsort = np.argsort(lengths)
    all_nodes = set()
    for l in lens:
        all_nodes.update(l)
    all_nodes = np.array(list(all_nodes))
    # sharp_pts = mesh_obj.vertices[all_nodes]
    centers = np.array([np.mean(mesh_obj.vertices[list(ppts)],axis=0) for ppts in lens])

    # SKELETONIZE - if we are just looking for general errors, not errors at endpoints, this can be skipped
    skel_mp = skeletonize.skeletonize_mesh(trimesh_io.Mesh(mesh_obj.vertices, 
                                            mesh_obj.faces),
                                            invalidation_d=20000,
                                            shape_function='cone',
                                            collapse_function='branch',
#                                             soma_radius = soma_radius,
                                            soma_pt=center,
                                            smooth_neighborhood=5,
#                                                     collapse_params = {'dynamic_threshold':True}
                                            )
    print("Skel done")
    # Process the skeleton to get the endpoints
    edges = skel_mp.edges.copy()

    edges_flat  = edges.flatten()
    edge_bins = np.bincount(edges_flat) 

    bps = np.squeeze(np.argwhere(edge_bins==3))
    eps = np.squeeze(np.argwhere(edge_bins==1))
    eps_nm = skel_mp.vertices[eps]

    eps_comp = distance_matrix(eps_nm, eps_nm)
    eps_comp[eps_comp == 0] = np.inf
    eps_thresh = np.argwhere(~(np.min(eps_comp, axis=0) < 3000))

    eps = np.squeeze(eps[eps_thresh])
    eps_nm = np.squeeze(eps_nm[eps_thresh])

    path_to_root_dict = {}
    for ep in eps:
        path_to_root_dict[ep] = skel_mp.path_to_root(ep)
        
    dists_defects = np.zeros(centers.shape[0])
    sizes = np.zeros(centers.shape[0])
    mesh_map = skel_mp.mesh_to_skel_map
    closest_skel_pts = mesh_map[[l[0] for l in lens]]
    dist_matrix = distance_matrix(centers, eps_nm)
    ct = 0

    closest_tip = np.zeros((centers.shape[0]))

    for center in tqdm(centers):
    #     skel_pts_dists = np.linalg.norm(skel_mp.vertices - center, axis=1)
    #     ep_pts_dists = np.linalg.norm(eps_nm - center, axis=1)
        
        closest_skel_pt = closest_skel_pts[ct]
        min_ep = np.inf
        eps_hit = []
        for j, ep in enumerate(eps):
            if closest_skel_pt in path_to_root_dict[ep]:
                eps_hit.append(j)
        if len(eps_hit) == 0:
            dists_defects[ct] = np.inf
            sizes[ct] = np.inf
            ct+=1
            continue
        
        dists = dist_matrix[ct, eps_hit]
    #     print(dists, eps_hit, center / [4,4,40])
        
        amin = np.argmin(dists)
        tip_hit = eps_hit[amin]
        min_dist = dists[amin]
        
        closest_tip[ct] = tip_hit
    #     print(np.argmin(ep_pts_dists), ep_found, eps_nm[np.argmin(ep_pts_dists)]/[4,4,40], eps_nm[j]/[4,4,40], center/[4,4,40])
        dists_defects[ct] = min_dist
        sizes[ct] = len(lens[ct])
        ct+=1
    dists_defects_sub = dists_defects[dists_defects < np.inf]
    sizes_sub = sizes[dists_defects < np.inf]
    centers_sub = centers[dists_defects < np.inf]
    tips_hit_sub = closest_tip[dists_defects < np.inf]
    closest_skel_pts_sub = closest_skel_pts[dists_defects < np.inf]
    inds_sub = np.arange(centers.shape[0])[dists_defects < np.inf]

    locs = np.argwhere(mesh_obj.facets_area > 50000)

    mesh_coords = mesh_obj.vertices[mesh_obj.faces]
    mean_locs = []
    mesh_ind = []
    fs = []
    for l in tqdm(locs):
        fs.append(np.sum(mesh_obj.facets_area[l]))
        fc = mesh_obj.facets[l[0]]
        vert_locs = mesh_coords[fc]
        mean_locs.append(np.mean(vert_locs[:, 0], axis=0))
        mesh_ind.append(fc[0])
    mesh_ind = mesh_obj.faces[mesh_ind][:, 0]
    mean_locs = np.array(mean_locs)
    dists_defects_facets = np.zeros(mean_locs.shape[0])
    mesh_map_facets = skel_mp.mesh_to_skel_map
    closest_skel_pts_facets = mesh_map[[m for m in mesh_ind]]
    dist_matrix_facets = distance_matrix(mean_locs, eps_nm)
    ct = 0

    closest_tip_facets = np.zeros((mean_locs.shape[0]))


    for center in tqdm(mean_locs):

        closest_skel_pt = closest_skel_pts_facets[ct]
        eps_hit = []
        for j, ep in enumerate(eps):
            if closest_skel_pt in path_to_root_dict[ep]:
                eps_hit.append(j)
        if len(eps_hit) == 0:
            dists_defects_facets[ct] = np.inf
            ct+=1
            continue
        
        dists = dist_matrix_facets[ct, eps_hit]
        
        amin = np.argmin(dists)
        tip_hit = eps_hit[amin]
        min_dist = dists[amin]
        
        closest_tip_facets[ct] = tip_hit
        dists_defects_facets[ct] = min_dist
        ct+=1
    dists_defects_sub_facets = dists_defects_facets[dists_defects_facets < np.inf]
    sizes_sub_facets = np.array(fs)[dists_defects_facets < np.inf]
    mean_locs_facets = mean_locs[dists_defects_facets < np.inf]
    tips_hit_sub_facets = closest_tip_facets[dists_defects_facets < np.inf]
    closest_skel_pts_sub_facets = closest_skel_pts_facets[dists_defects_facets < np.inf]
    inds_sub_facets = np.arange(mean_locs.shape[0])[dists_defects_facets < np.inf]
    ranks_ep_facets = sizes_sub_facets**2 / dists_defects_sub_facets
    #ranks_ep_facets_filt = ranks_ep_facets[ranks_ep_facets > 2e7]
    mean_locs_send_facets = mean_locs_facets[np.argsort(ranks_ep_facets)][::-1][:20]
    final_mask_facets = np.full(mean_locs_send_facets.shape[0], True)
    tips_hit_send_facets = tips_hit_sub_facets[np.argsort(ranks_ep_facets)][::-1][:20]
    uns, nums = np.unique(tips_hit_send_facets, return_counts=True)

    for un, num in zip(uns, nums):
        if num > 1:
            final_mask_facets[np.argwhere(tips_hit_send_facets == un)[1:]] = False
    # Also ranking each component based on its PCA- if the first component is big enough, the points are mostly linear
    # These point sets seem to be less likely to be true errors
    from sklearn.decomposition import PCA
    pca_vec = np.zeros(inds_sub.shape[0])
    for i in range(inds_sub.shape[0]):
        pca = PCA()#n_components=2)
        pca.fit(mesh_obj.vertices[lens[inds_sub[i]]])

        pca_vec[i] = pca.explained_variance_ratio_[0]

    dists_defects_sub[dists_defects_sub < 4000] = 100
    dists_defects_norm = dists_defects_sub #/ np.max(dists_defects_sub)
    ranks_ep = sizes_sub / dists_defects_norm * (1-pca_vec)
    ranks = sizes_sub**2 * (1-pca_vec)

    #ranks_ep_errors_filt = ranks_ep[ranks_ep > .1]
    centers_ep_send_errors = centers_sub[np.argsort(ranks_ep)][::-1][:20]
    final_mask_eps = np.full(centers_ep_send_errors.shape[0], True)
    tips_hit_send_ep = tips_hit_sub[np.argsort(ranks_ep)][::-1][:20]
    uns, nums = np.unique(tips_hit_send_ep, return_counts=True)

    for un, num in zip(uns, nums):
        if num > 1:
            final_mask_eps[np.argwhere(tips_hit_send_ep == un)[1:]] = False
    centers_errors_ep = centers_ep_send_errors[final_mask_eps]
    centers_errors = centers_sub[np.argsort(ranks)[::-1]][:20]

    #if len(centers_errors.shape) > 1 and centers_errors.shape[0] > 0 and len(centers_errors_ep.shape) > 1 and len(centers_errors_ep.shape[0] > 0):
    #    centers_errors = centers_errors[np.min(distance_matrix(centers_errors, centers_errors_ep), axis=1)>1000]

    facets_send_final = mean_locs_send_facets[final_mask_facets] / [4,4,40]
    errors_send = centers_errors / [4,4,40]
    errors_tips_send = centers_errors_ep / [4,4,40]
    encapsulated_centers = [e[0] for e in encapsulated_ids]
    encapsulated_lens = [e[1] for e in encapsulated_ids]
    sorted_encapsulated_send = np.array(encapsulated_centers)[np.argsort(encapsulated_lens)][::-1]
    return sorted_encapsulated_send, facets_send_final, errors_send, errors_tips_send, np.squeeze(ranks_ep_facets[[np.argsort(ranks_ep_facets)][::-1][:20]]), np.squeeze(ranks[np.argsort(ranks)[::-1]][:20]), np.squeeze(ranks_ep[np.argsort(ranks_ep)][::-1][:20])


def return_filled(root_id, loc):
    from agents import data_loader
    import fill_voids
    client = CAVEclient('minnie65_phase3_v1')

    radius = (200,200,15)
    root_id = 864691135104596813
    center = loc[0]/2, loc[1]/2, loc[2]

    seg = np.squeeze(data_loader.get_seg(center[0] - radius[0],
                center[0] + radius[0],
                center[1] - radius[1],
                center[1] + radius[1],
                center[2] - radius[2],
                center[2] + radius[2]))
    u, ns = np.unique(seg, return_counts=True)

    u = u[np.argsort(ns)][::-1]
        
    seg_ids = np.array(u).astype(np.int64)
    root_id_newest = client.chunkedgraph.get_latest_roots(root_id)

    if not root_id_newest in seg_ids:
        #print('int')
        out_of_date_mask = client.chunkedgraph.is_latest_roots(seg_ids)
        seg_ids = np.array(seg_ids)
        out_of_date_seg_ids = seg_ids[~(out_of_date_mask)]
        seg_dict={}

        for s in out_of_date_seg_ids:
         #   print(s)
            if s == 0:
                seg_dict[0] = 0
                continue
                
            new_s = client.chunkedgraph.get_latest_roots(s)
            
            if root_id in new_s:
                seg_mask = s
                break
                
            seg_ids[seg_ids == s] = new_s[-1]
            seg_dict[new_s[-1]]=s
        for s in seg_ids[out_of_date_mask]:
            seg_dict[s] = s
    mask = seg == seg_mask
    filled = fill_voids.fill(mask)
    diff = ~mask * filled
    return list(np.unique(seg[diff]))
