from distutils.log import error
from logging import root
import backoff
import numpy as np
from scipy.ndimage.filters import convolve
import pandas as pd
from caveclient import CAVEclient
import cv2 
import datetime
from sofima.flow_field import JAXMaskedXCorrWithStatsCalculator
from sofima.flow_utils import clean_flow
from sofima.warp import ndimage_warp
from sofima import mesh
from sofima import map_utils
from connectomics.common import bounding_box
from sofima import flow_utils

def load_membrane_vectors(precomp_fn):
    # If there already is a saved out precompute file, just load it!
    temp = np.load(precomp_fn)
    data = temp["data"]
    return data

def precompute_membrane_vectors(mem, memz, mem_radius=3):
    """
    Convolve membrane with each directional sensor and then
    concatenate channels to form a lookup table
    This will be equivalent to a sensor sampling within a given radius
    and computing point-wise influence vectors
    :param mem: data matrix (edge filtered)
    :param memz: data martix, edge filtered along z axis
    :param comp_flag: Whether or not to precompute or load
    :param precomp_fn: location of saved-out precompute file
    :return:
    """

    rad = int(mem_radius / 2)
    a = np.arange(mem_radius) - rad
    a_s = np.arange(3) - rad
    a[:rad], a[-rad:] = np.flip(a[:rad]), np.flip(a[-rad:])
    a_s[:rad], a_s[-rad:] = np.flip(a_s[:rad]), np.flip(a_s[-rad:])
    kx, ky, kz = np.meshgrid(a, a, a_s)
    # print("KS", kx.shape, ky.shape, kz.shape)
    mem_x = convolve(mem.astype(int), kx, mode="constant", cval=5)[..., np.newaxis]
    mem_y = convolve(mem.astype(int), ky, mode="constant", cval=5)[..., np.newaxis]
    # Note memz is used here - the yz plane edge detection is important
    mem_z = convolve(memz.astype(int), kz, mode="constant", cval=5)[..., np.newaxis]

    data = np.concatenate((mem_x, mem_y, mem_z), axis=-1)

    return data

def position_merge(ep, root_id, merge_d, endpoint_nm, n_errors, mean_err, max_err, seg_remap):
    merge_df = pd.DataFrame()
    import time
    ms = set()
    merges = {}

    for k, v in merge_d.items():
        merges[(seg_remap[int(k[0])], seg_remap[int(k[1])])] = v

    for m in merges:
        ms.add(int(m[0]))
        ms.add(int(m[1]))

    m_list = list(ms)
    if len(m_list) > 0:
        tic = time.time()
        # soma_table = get_soma(m_list)
        toc = time.time()
        print("soma time", toc - tic)
        polarity_table = is_dendrite(endpoint_nm, m_list)
        tic = time.time()
        print("polarity time", tic - toc)
        # thresholds_dict, thresholds_list = trajectory_filter(root_id, m_list, seg)
        toc = time.time()
        # print("trajectory time", toc - tic)        
        root_id = [int(x) for x in root_id]
        for i, k in enumerate(merges):
            soma_filter = False
            if int(k[0]) in root_id:
                extension = int(k[1])
                weight = int(merges[k])
            elif int(k[1]) in root_id:
                extension = int(k[0])
                weight = float(merges[k])
            else:
                continue
            if extension == 0:
                continue
            # if soma_table.get_soma(extension) > 0:
            #     soma_filter=True
            mixed_polarity_root, n_pre_root, n_post_root, n_pre_seg, n_post_seg, mixed_polarity_seg = polarity_table.dendrite_filt(root_id[0], extension)

            d = {
                "EP":[ep], 
                "Root_id":str([root_id[0]]), 
                "Extension":str(extension)+"/", 
                "Weight":weight, 
                "mix_root": mixed_polarity_root, 
                "n_pre_root":n_pre_root, 
                "n_post_root":n_post_root, 
                "n_pre_seg":n_pre_seg, 
                "n_post_seg":n_post_seg,
                "mix_seg":mixed_polarity_seg,
                "Soma_filter":soma_filter,
                "Error_num":n_errors,
                "Error_max":max_err,
                "Error_mean":mean_err,
                # "Angle":thresholds_dict[int(extension)]
                }
            merge_df = pd.concat([merge_df, pd.DataFrame(d, index=[i])])
        print("processing time", time.time()-toc)
    return merge_df

def find_center(seg, seg_id):
    segmentation_coords = np.array(np.argwhere(seg == seg_id))
    return np.mean(segmentation_coords, axis=1)

def create_post_matrix(pos_histories, seg, data_shape,merge=False,threshold=5):
    pos_matrix = np.full(data_shape, np.nan)
    merge_d = {}
    for i, h in enumerate(pos_histories):
        p0 = h[0]
        p0 = p0.astype(int)
        init_rid = seg[p0[0], p0[1], p0[2]]
        added = set()
        for j, p in enumerate(h):
            p = p.astype(int)
            if p[2] > data_shape[2] or p[2] < 0:
                continue
            seg_id = seg[p[0], p[1], p[2]]
            if merge and seg_id != init_rid:
                # if seg_id not in added:
                try:
                    merge_d[(str(init_rid), str(seg_id))] += 1
                except:
                    merge_d[(str(init_rid), str(seg_id))] = 1
                added.add(seg_id)
            pos_matrix[p[0], p[1],p[2]] = j#init_rid
    threshold_merges = {}
    for k in merge_d:
        v = merge_d[k]
        if v > threshold:
            threshold_merges[k] = v
    return pos_matrix, merge_d

def remove_keymap_conflicts(new_keys_set):
    import matplotlib.pyplot as plt
    """
    Removes matplotlib in-built keybinding conflicts
    :param new_keys_set: dict of chars to be unbound
    :return:
    """
    for prop in plt.rcParams:
        if prop.startswith("keymap."):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)


def multi_slice_viewer(membranes, positions=None, start_slice=-1):
    """
    View 3d volume one slice at a time
    :param membranes: 3D numpy array of greyscale data
    :param positions: [Optional] Pointcloud of agent positions
    :return:
    """
    import matplotlib.pyplot as plt

    if positions is None:
        positions = np.zeros_like(membranes)

    remove_keymap_conflicts({"j", "k"})
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.paths = positions
    ax.mems = membranes

    if start_slice == -1:
        ax.index = membranes.shape[2] // 2
    else:
        ax.index = start_slice
    # make the colormaps
    mem_colors = plt.get_cmap("Greys")
    agent_colors = plt.get_cmap("tab20")

    # Initialize color maps to eliminate background color
    agent_colors._init()

    # create your alpha array and fill the colormap with them.
    agent_colors._lut[0, -1] = 0
    # print("A", ax.index)
    ax.imshow(
        membranes[:, :, ax.index].T, interpolation="nearest", cmap=mem_colors, origin="upper"
    )
    ax.imshow(
        positions[:, :, ax.index].T, interpolation="nearest", cmap=agent_colors, origin="upper"
    )

    ax.set_title(start_slice)
    fig.canvas.mpl_connect("key_press_event", process_key)
    # fig.canvas.mpl_connect('scroll_event', process_key)

def process_key(event):
    """
    Action to perform once the 'up' or 'down' key is pressed
    :param event: action of pushing the key, processed by fig.canvas.mpl_connect()
    :return:
    """
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == "j":
        move_slice(ax, -1)

    elif event.key == "k":
        move_slice(ax, 1)

    if event.button == "up":
        move_slice(ax, 1)
    elif event.button == "down":
        move_slice(ax, -1)
    fig.canvas.draw()

def move_slice(ax, sign):
    """
    Go to the next or previous slice
    :param ax: Which axis of plt object to move
    :param sign: Which direction to move the volume
    :return:
    """
    ax.index = (ax.index + sign) % ax.paths.shape[2]  # wrap around using %
    ax.images[0].set_array(ax.mems[:, :, ax.index].T)
    ax.images[1].set_array(ax.paths[:, :, ax.index].T)
    ax.set_title(ax.index)

class get_soma():
    @backoff.on_exception(backoff.expo, Exception, max_tries=8)
    def __init__(self, seg_ids):
        self.seg_ids = seg_ids
        cave_client = CAVEclient('minnie65_phase3_v1')
        soma = cave_client.materialize.query_table(
            "nucleus_neuron_svm",
            filter_in_dict={'pt_root_id':seg_ids},
            select_columns=['id','pt_root_id', 'pt_position']
        )
        soma_dict = {}
        for s in seg_ids:
            soma_dict[int(s)] = soma[soma['pt_root_id'] == s].shape[0]
        self.soma_dict = soma_dict

    def get_soma(self, seg_id):
        return self.soma_dict[seg_id]

def make_bounding_box(point, distance):

    point_a = [coord - distance for coord in point]
    point_b = [coord + distance for coord in point]

    return [point_a, point_b]

class is_dendrite():
    @backoff.on_exception(backoff.expo, Exception, max_tries=8)
    def __init__(self, endpoint, seg_ids, update=False):
        seg_ids = [int(x) for x in seg_ids if int(x) != 0] 
        if not update:
            new_ids = seg_ids
            seg_dict = dict(zip(seg_ids, seg_ids))
        # Create a client for the Minnie65 PCG and Tables
        client = CAVEclient('minnie65_phase3_v1')
        bounding_box = make_bounding_box(endpoint, 2500)
        synapse_table = 'synapses_pni_2'
        # print(new_ids)
        post_ids = client.materialize.query_table(synapse_table,
            filter_in_dict={'post_pt_root_id': new_ids},
            filter_spatial_dict={'post_pt_position': bounding_box},
            select_columns=['post_pt_root_id','post_pt_position'],
            timestamp = datetime.datetime(2022, 4, 29)
        )

        pre_ids = client.materialize.query_table(synapse_table,
            filter_in_dict={'pre_pt_root_id': new_ids},
            filter_spatial_dict={'pre_pt_position': bounding_box},
            select_columns=['pre_pt_root_id','pre_pt_position'],
            timestamp = datetime.datetime(2022, 4, 29)
        )
        dendrite_df = pd.DataFrame()
        for ind, i in enumerate(new_ids):
            len_post = len(post_ids[post_ids["post_pt_root_id"] == i])
            len_pre = len(pre_ids[pre_ids["pre_pt_root_id"] == i])

            if len_post > 0 and len_pre > 0:
                mixed_polarity=True
            else:
                mixed_polarity=False
            dendrite_df = pd.concat([dendrite_df, pd.DataFrame({'seg_id':seg_dict[i], 'n_pre':len_pre, 'n_post':len_post, 'mixed_polarity':mixed_polarity}, index=[ind])])
        self.current_ids = seg_dict
        self.dendrite_df = dendrite_df

    def dendrite_filt(self, root_id, seg_id):
        root = self.dendrite_df[self.dendrite_df.seg_id==root_id].iloc[0]
        mixed_polarity_root = root.mixed_polarity
        n_pre_root = root.n_pre
        n_post_root = root.n_post
        
        seg = self.dendrite_df[self.dendrite_df.seg_id==seg_id].iloc[0]

        n_pre_seg = seg.n_pre
        n_post_seg = seg.n_post
        mixed_polarity_seg = seg.mixed_polarity
    
        return mixed_polarity_root, n_pre_root, n_post_root, n_pre_seg, n_post_seg, mixed_polarity_seg

def shift_detect(em, radius, n_windows, n_hits=-1,threshold=10,zero_threshold=.75):
    if n_hits == -1:
        if n_windows > 1:
            n_hits = n_windows - 1
        else:
            n_hits=1
    em_shape = em.shape
    centers = np.linspace(0,em_shape[0],n_windows + 2).astype(int)[1:-1]
    errors = np.full(em_shape[2], True)
    errors_zero = np.full(em_shape[2], False)
    error_dict = {}

    if np.sum(em[:,:,0]==0) > zero_threshold*em_shape[0]*em_shape[1]:
        errors_zero[0] = True
    for i in range(1,em.shape[2]):
        means = []
        locs = []
        last_frame = em[:,:,i-1]
        frame=em[:,:,i]

        for k in range(n_windows):
            if centers[k]-radius < 0 or centers[k]+radius > em_shape[0]:
                continue
            xmin, xmax = centers[k]-radius, centers[k]+radius
            
            last_frame_center = last_frame[xmin:xmax, xmin:xmax]
            res = cv2.matchTemplate(frame, last_frame_center, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            mse = np.sqrt(np.sum(np.square(np.array(min_loc)-np.array((centers[k]-radius,centers[k]-radius)))))
            means.append(mse)
            locs.append(min_loc)
        if np.sum(np.array(means) < threshold) >= n_hits:
            errors[i-1] = False
            errors[i] = False
            error_dict[i] = "No Error"
            error_dict[i-1] = "No Error"
        else:
            error_dict[i] = "Shift Error"
            error_dict[i-1] = "Shift Error"
        if np.sum(em[:,:,i]==0) > zero_threshold*em_shape[0]*em_shape[1]:
            errors_zero[i] = True
            error_dict[i] = "Zero"

    return errors, errors_zero, error_dict

def detect_error_locs(em, threshold=.98, zero_threshold=.5):

    em_center = np.array([em.shape[0]//2, em.shape[0]//2])   
    radius = int(em.shape[0]//2)
    xmin, xmax = em_center[0]-radius, em_center[0]+radius

    em_shape = em.shape
    errors = np.zeros(em_shape[2])
    error_dict = {}
    still_good_flag = True
    
    # Find first good slice
    start = 1
    while True:
        last_frame = em[:,:,start-1]
        if np.sum(last_frame == 0) / np.product(em.shape[:2]) > zero_threshold:
            start += 1
        else:
            frame=em[:,:,start]
            last_frame_center = last_frame[xmin:xmax, xmin:xmax]

            if cv2.minMaxLoc(cv2.matchTemplate(frame, last_frame_center, cv2.TM_CCORR_NORMED))[1] > threshold:
                break
            else:
                start+=1
        
    for i in range(start,em.shape[2]-1):

        if still_good_flag:
            last_frame = em[:,:,i-1]
        frame=em[:,:,i]

        last_frame_center = last_frame[xmin:xmax, xmin:xmax]
        
        res = cv2.matchTemplate(frame, last_frame_center, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val < threshold:
            still_good_flag = False
            errors[i] += 1
        else:
            still_good_flag = True

    error_dict = {i:errors[i] for i in range(errors.shape[0])}
    return errors, error_dict

def remove_shifts(em, errors, zero_or_remove='zero'):
    if not errors.dtype == bool:
        errors = errors > 0
    if zero_or_remove == 'zero':
        em[:,:,errors] = 0
    elif zero_or_remove=='remove':
        em = em[:,:,~errors]
    return em
    
def get_contacts(seg, root_id,conv_size=3):
    from scipy.ndimage.filters import convolve
    import copy
    import agents.data_loader as data_loader

    seg_mask = copy.deepcopy(seg)
    seg_copy = copy.deepcopy(seg)

    seg_mask[np.logical_not(seg_mask == root_id)] = 0
    seg_mask[seg_mask > 0] = 1
    kernel = np.ones((conv_size,conv_size,conv_size), np.uint8)
    seg_dilate = convolve(seg_mask, kernel, mode="constant", cval=0)

    close_ids = np.unique(seg_copy[seg_dilate > 0])
    all_ids = np.unique(seg_copy)

    centers = {}
    sizes = {}

    points = np.argwhere(seg_copy == root_id)
    points_z = points[:,2]
    centers_list = []
    sizes_list = []
    for z in range(np.min(points_z), np.max(points_z)+1):
        if z < 2 or z > 198:
            continue
        points_slice = np.argwhere(seg_copy[:,:,z] == root_id)
        centers_list.append([*list(np.mean(points_slice, axis=0).astype(int)), z])
        sizes_list.append(len(points_slice))
    centers[root_id] = centers_list
    sizes[root_id] = sizes_list
    for rid in close_ids:
        points = np.argwhere(seg_copy == rid)
        points_z = points[:,2]
        centers_list = []
        sizes_list = []
        for z in range(np.min(points_z), np.max(points_z)+1):
            points_slice = np.argwhere(seg_copy[:,:,z] == rid)
            mean_pt = np.mean(points_slice, axis=0)
            if points_slice.shape[0] > 0 and seg[int(mean_pt[0]), int(mean_pt[1]), z] == rid:
                centers_list.append([*list(mean_pt.astype(np.uint64)), z])
                sizes_list.append(len(points_slice))

        centers[rid] = centers_list
        sizes[rid] = sizes_list

    return close_ids, all_ids, centers, sizes

def get_public_seg_ids(seg_ids):
    if type(seg_ids) != list:
        seg_ids = [seg_ids]
    client = CAVEclient("minnie65_phase3_v1")
    ts = client.materialize.get_timestamp(117)
    past_ids = client.chunkedgraph.get_past_ids(root_ids=seg_ids,timestamp_past=ts)
    ids = list(past_ids['past_id_map'].keys())
    public_ids = []
    seg_dict={}

    for j, id_key in enumerate(ids):
        k = past_ids['past_id_map'][id_key]
        public_ids.append(int(k[0]))

    return public_ids

def get_current_seg_ids(seg_ids):
    if type(seg_ids) != list:
        try:
            seg_ids = list(seg_ids)
        except TypeError:
            seg_ids = [seg_ids]
    client = CAVEclient("minnie65_phase3_v1")
    out_of_date_mask = client.chunkedgraph.is_latest_roots(seg_ids)
    seg_ids = np.array(seg_ids)
    out_of_date_seg_ids = seg_ids[np.logical_not(out_of_date_mask)]
    seg_dict={}
    for s in out_of_date_seg_ids:
        new_s = client.chunkedgraph.get_latest_roots(s)
        seg_ids[seg_ids == s] = new_s[-1]
        seg_dict[new_s[-1]]=s
    for s in seg_ids[out_of_date_mask]:
        seg_dict[s] = s
    return seg_ids, seg_dict


def trajectory_filter(root_id, seg_ids, seg):
    angle_dict = {}
    angle_list = []
    if type(root_id) == list:
        root_id = root_id[0]
    meangrad_root = calc_seg_gradient(int(root_id), seg)
    for i in seg_ids:
        if int(i) == int(root_id):
            continue
        meangrad_seg = calc_seg_gradient(i, seg)
        dot = np.dot(meangrad_root, meangrad_seg)
        angle = np.arccos(dot)
        angle_dict[int(i)]=angle
        angle_list.append(angle)
    return angle_dict, angle_list

def calc_seg_gradient(seg_id, seg, rez = np.array([4,4,40])):
    import cc3d

    seg_mask = seg == int(seg_id)
    seg_dilate = cv2.dilate(seg_mask.astype(np.uint8), np.ones((5,5)), iterations=1)

    labels_out = cc3d.largest_k(seg_dilate, k=1, connectivity=6, delta=0)

    coords = np.argwhere(seg_mask * labels_out) #* rez
    points_z = coords[:,2]
    unique_pointz = sorted(np.unique(points_z))
    gradients = np.zeros((len(unique_pointz),3))

    # Weight slices closer to the midpoint more heavily
    slice_weights = np.zeros(seg.shape[2])
    for i in range(seg.shape[2]):
        diff = abs(i - seg.shape[2]/2-1) + 1
        slice_weights[i] = 1/diff

    for i, z in enumerate(range(1,len(unique_pointz))):
        cz, lz = unique_pointz[z], unique_pointz[z-1]
        
        pointz_last = coords[coords[:,2] == lz]
        pointz = coords[coords[:,2] == cz]
        mean_slice = np.mean(pointz[:,:2], axis=0) 
        mean_last = np.mean(pointz_last[:,:2], axis=0)

        mean_slice=np.mean(pointz[:,:2], axis=0) 
        mean_last=np.mean(pointz_last[:,:2], axis=0)
        
        meanshift = mean_slice - mean_last
        gradients[i] = [meanshift[0], meanshift[1], 0]
        gradients[i][2] = 1
        gradients[i] = gradients[i] * slice_weights[z]

    meangrad = np.mean(gradients, axis=0)
    meangrad = meangrad / np.linalg.norm(meangrad)
    return meangrad
        
from scipy.ndimage import gaussian_filter
def shift_detect_rearrange(em_o, seg, radius, zero_threshold=.7, noise_sigma=0, z_stop=True):
    em=em_o

    if noise_sigma > 0:
        em = np.zeros_like(em_o)
        for i in range(em_o.shape[2]):
            em[:, :, i] = gaussian_filter(em_o[:, :, i], sigma=noise_sigma)
    else:
        em = em_o
    em_shape = em.shape
    em_middle = np.array(em.shape[0]//2, em.shape[1]//2)
    em_shape_z_half = em_shape[2] // 2 - 1
    
    pixels_per_frame = em_shape[0]*em_shape[1]
    
    seg_locs = np.argwhere(seg[:, :, em_shape_z_half])
    
    em_spawn = em_shape_z_half
    m = 1
    d = 1
    em_flag = True
    ct=0
    # finding the last slide 
    while len(seg_locs) < 1 or em_flag:
        em_spawn = em_shape_z_half + m*d
        seg_locs = np.argwhere(seg[:, :, em_spawn])
        if m > em_shape_z_half - 2:
            mean_loc_seg = np.array([em_shape[0]//2, em_shape[1]//2])
        if len(seg_locs) > 0:
            mean_loc_seg = np.mean(seg_locs, axis=0).astype(int)

            mean_loc_seg += (em.shape[0] - seg.shape[0]) // 2
            xmin, xmax = mean_loc_seg[0] - radius, mean_loc_seg[0] + radius
            ymin, ymax = mean_loc_seg[1] - radius, mean_loc_seg[1] + radius
            last_frame_template = em[xmin:xmax, ymin:ymax, em_spawn]
            
            if np.sum(last_frame_template < 10) > zero_threshold*(2*radius)**2:
                em_flag = True
            else:
                em_flag = False
        d = d * -1
        ct+=1
        if ct % 2 == 0:
            m+=1
        
    em_shape_z_half = em_spawn

    n_total =  np.sum(seg > 0)
    n_above = np.sum(seg[:,:,:em_shape_z_half] > 0)
    n_below = np.sum(seg[:,:,em_shape_z_half:] > 0)
    print("N above below", n_above, n_below, n_total)
    if n_above > .05*n_total:
        go_below = True
    if n_below > .05*n_total:
        go_above = True

    dropped = []
    shifts = {}
    shifts[em_shape_z_half] = np.array([mean_loc_seg[0], mean_loc_seg[1]])
    shifts[em_shape_z_half] = shifts[em_shape_z_half] - np.array([em_shape[0]//2, em_shape[1]//2])
    if (not z_stop) and go_above: 
        for i in range(em_shape_z_half + 1, em.shape[2]):
            current_frame = em[:, :, i]
            if np.sum(current_frame < 10) > zero_threshold*pixels_per_frame:
                # print(i, "drop", np.sum(current_frame < 10) / pixels_per_frame)
                dropped.append(i)
                shifts[i] = np.array([np.nan, np.nan])
                continue
            res = cv2.matchTemplate(current_frame, last_frame_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)        
            if max_val < .1:
                # print("Warp", i)
                shifts[i] = np.array([np.nan, np.nan])
                dropped.append(i)
                continue
            else:
                max_loc_dist = np.linalg.norm(em_middle - max_loc)
                # print(max_loc_dist, i)
            min_loc = max_loc
            min_loc = min_loc[1], min_loc[0]
            shifts[i] = np.array(min_loc) + radius
            shifts[i] = shifts[i] - np.array([em_shape[0]//2, em_shape[1]//2])

            # print(i, min_loc, max_val, shifts[i])
            last_frame_template = current_frame[min_loc[0]:min_loc[0] + 2*radius, min_loc[1]:min_loc[1] + 2*radius]
            
    # down
    last_mean = np.array([em_shape[0], em_shape[1]])
    
    xmin, xmax = mean_loc_seg[0] - radius, mean_loc_seg[0] + radius
    ymin, ymax = mean_loc_seg[1] - radius, mean_loc_seg[1] + radius
    last_frame_template = em[xmin:xmax, ymin:ymax, em_shape_z_half]
    
    if (not z_stop) and go_below:
        for i in range(em_shape_z_half-1, -1, -1):
            current_frame = em[:, :, i]
            if np.sum(current_frame < 10) > zero_threshold*pixels_per_frame:
                # print(i, "drop", np.sum(current_frame < 10) / pixels_per_frame)
                dropped.append(i)
                shifts[i] = np.array([np.nan, np.nan])
                continue
            res = cv2.matchTemplate(current_frame, last_frame_template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)        
            if max_val < .1:
                # print("Warp", i)
                shifts[i] = np.array([np.nan, np.nan])
                dropped.append(i)
                continue
            min_loc = max_loc
            min_loc = min_loc[1], min_loc[0]
            shifts[i] = np.array(min_loc) + radius
            shifts[i] = shifts[i] - np.array([em_shape[0]//2, em_shape[1]//2])
            # print(i, min_loc, max_val, shifts[i])

            last_frame_template = current_frame[min_loc[0]:min_loc[0] + 2*radius, min_loc[1]:min_loc[1] + 2*radius]
        
    return dropped, shifts

def restitch(vol, radius, shift):
    restitched = np.zeros((2*radius, 2*radius, vol.shape[2]), dtype=vol.dtype)
    ct = 0
    for k in shift:
        if np.isnan(shift[k][0]):
            continue
        mv = shift[k]

        em_slice = vol[:,:,k]
        em_pad = np.zeros((em_slice.shape[0]+2*radius, em_slice.shape[1]+2*radius), dtype=vol.dtype)
        em_pad[radius:-radius, radius:-radius] = em_slice

        x1,x2 = mv[0] - radius + em_pad.shape[0]//2, mv[0] + radius + em_pad.shape[0]//2
        y1,y2 = mv[1] - radius + em_pad.shape[1]//2, mv[1] + radius + em_pad.shape[1]//2
        restitched[:, :, ct] = em_pad[x1:x2, y1:y2]
        ct += 1

    restitched = restitched[:, :, :ct]
    return restitched, vol.shape[2] - ct
    
def em_preprocess(em1):
    em = em1.copy().astype(np.float32)
    for i in range(em.shape[2]):
        sli = em[...,i]
        sli_nz = sli[sli > 0]
        sli = (sli - np.mean(sli_nz)) / np.std(sli_nz)
        sli = ndimage.gaussian_filter(sli, 3)
        em[...,i]=sli
    return em

def detect_artifacts_template(em_o, errors_zero,diam=30):
    
    em1 = em_o.copy()
    # Then, start at that slice and direction and work your way up or down and track
    # the first place where seg cuts out
    em_center = np.array([em1.shape[0]//2, em1.shape[0]//2])
    radius = int((em1.shape[0]//4))
    xmin, xmax = em_center[0]-radius, em_center[0]+radius
    # Making Gaussian to template match
    # Radius of Gaussian at 10 works well experimentally
    x = np.arange(0, diam, 1, float)
    center = ((diam-1)/2,(diam-1)/2)
    # Gaussian scalar constant
    fwhm = 15
    y = x[:,np.newaxis]
    mask = np.exp(-4*np.log(2) * ((x-center[0])**2 + (y-center[1])**2) / fwhm**2)
    mask = 2*(mask-.5)
    mask = mask / np.max(mask)

    errors_zero_template = np.zeros(em1.shape[2])
    good_locs = np.ones(em1.shape[2])
    # Find first good slice
    start = 1
    while True:
        if not errors_zero[start]:
            break
        else:
            start+=1
            
    zero = False
    for i in range(start,em1.shape[2]-1):
        if errors_zero[i]:
            errors_zero_template[i] = 1
            if not zero:
                start = i-1
                zero = True
            continue

        if zero:
            indx = start
            zero=False
        else:
            indx = i-1

        last_frame = em1[:,:,indx]
        frame=em1[:,:,i]
        last_frame_center = last_frame[xmin:xmax, xmin:xmax]
        
        res = cv2.matchTemplate(frame, last_frame_center, cv2.TM_CCOEFF_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res)

        mres = np.min(res)
        res = (res - mres)/ (np.max(res) - mres) if np.max(res) - mres != 0 else np.zeros_like(res)
        iden = cv2.matchTemplate(res.astype(np.float32), mask.astype(np.float32), cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(iden)
        if max_val < 30:
            errors_zero_template[i] = 1
            errors_zero_template[indx] = 1
        else:
            good_locs[i] = 0
            good_locs[indx] = 0
    errors_zero_template = np.logical_and(errors_zero_template, good_locs)

    return errors_zero_template

def get_warp(em, errors, patch_size = 256, stride = 128):
    err_pre_post = []

    i=0
    flag = False
    
    while i < errors.shape[0]:
        if errors[i] and flag:
            pre = i-1
            while True:
                i+=1
                if i == errors.shape[0]:
                    break
                if not errors[i]:
                    post = i
                    err_pre_post.append((pre,post))
                    break
        elif not errors[i]:
            flag = True
        i+=1
    
    maskcorr = JAXMaskedXCorrWithStatsCalculator()
    
    warp_dict_reg = {}
    warp_dict_clean = {}
    warp_dict_flow = {}
    
    for error in err_pre_post:

        flow_1 = maskcorr.flow_field(em[:,:,error[0]], em[:,:,error[1]], patch_size, stride)

        pad_y = patch_size // 2 // stride
        pad_x = patch_size // 2 // stride

        flow_pad_1 = np.pad(flow_1, [[0, 0], [pad_y, pad_y - 1], [pad_x, pad_x - 1]], constant_values=np.nan)

        kwargs = {"min_peak_ratio": 1.1, "min_peak_sharpness": 1.1, "max_deviation": 0, "max_magnitude": 0}


        ffield_clean_1 = clean_flow(flow_pad_1[:, np.newaxis, ...], **kwargs)

        config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                                        num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                        dt_max=100, prefer_orig_order=True,
                                        start_cap=0.1, final_cap=10)

        x = np.zeros_like(ffield_clean_1)
        x, _, _ = np.array(mesh.relax_mesh(x, ffield_clean_1, config))
        x = np.array(x)[:,0,...]
        
        warp_dict_reg[error] = x
        warp_dict_clean[error] = ffield_clean_1
        warp_dict_flow[error] = flow_1
    return warp_dict_reg, warp_dict_clean, warp_dict_flow

def detect_error_locs(em_o, threshold=16, diam=10):
    
    em1 = em_o.copy().astype(np.float32)
    # Then, start at that slice and direction and work your way up or down and track
    # the first place where seg cuts out
    em_center = np.array([em1.shape[0]//2, em1.shape[0]//2])

    # Making Gaussian to template match
    # Radius of Gaussian at 10 works well experimentally
    x = np.arange(0, diam, 1, float)
    center = ((diam-1)/2,(diam-1)/2)
    # Gaussian scalar constant
    fwhm = 15
    y = x[:,np.newaxis]
    mask = np.exp(-4*np.log(2) * ((x-center[0])**2 + (y-center[1])**2) / fwhm**2)
    
    radius = int(em1.shape[0]//4)
    xmin, xmax = em_center[0]-radius, em_center[0]+radius

    em_shape = em1.shape
    errors_linear = np.zeros(em_shape[2])
    still_good_flag = True
    linear_shifts = {}
    # Find first good slice
    start = 1
    while True:
        last_frame = em1[:,:,start-1]
        if np.sum(last_frame == 0) / np.product(em1.shape[:2]) > .5:
            start += 1
        else:
            frame=em1[:,:,start]
            last_frame_center = last_frame[xmin:xmax, xmin:xmax]

            if cv2.minMaxLoc(cv2.matchTemplate(frame, last_frame_center, cv2.TM_CCORR_NORMED))[1] > .5:
                break
            else:
                start+=1
    em = em1.copy()
    for i in range(em.shape[2]):
        sli = em[...,i]
        sli[sli < np.mean(sli)] = 0
        sli[sli >= np.mean(sli)] = 1
    
    for i in range(start,em.shape[2]-1):

        if still_good_flag:
            last_frame = em[:,:,i-1]
        frame=em[:,:,i]

        last_frame_center = last_frame[xmin:xmax, xmin:xmax]
        
        res = cv2.matchTemplate(frame, last_frame_center, cv2.TM_CCORR_NORMED)
        
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        max_loc1 = max_loc[1] - res.shape[0]//2, max_loc[0] - res.shape[1]//2
        dist = np.linalg.norm(max_loc1)
        if max_val < .5:
            continue
        else:
            still_good_flag = True
            if dist > threshold:
                mres = np.min(res)
                res = (res - mres)/ (np.max(res) - mres)

                iden = cv2.matchTemplate(res.astype(np.float32), mask.astype(np.float32), cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(iden)
                shift = max_loc[1] - iden.shape[0]//2, max_loc[0] - iden.shape[1]//2
                dist = np.linalg.norm(shift)

                if dist > threshold:
                    errors_linear[i]+=1
                    linear_shifts[(i-1, i)] = shift

    return errors_linear, linear_shifts

def shift_by_slice(em, shifts):
    shifts_total = np.zeros((em.shape[2],2))
    running_shift = np.zeros((1,2))
    
    for i in range(em.shape[2]//2 + 1, em.shape[2]):
        if (i-1, i) in shifts:
            running_shift += shifts[(i-1, i)]
        shifts_total[i] = running_shift
    running_shift = np.zeros((1,2))
    for i in range(em.shape[2]//2, 0, -1):
        if (i-1, i) in shifts:
            running_shift += shifts[(i-1, i)]
        shifts_total[i-1] = running_shift
    return shifts_total

def detect_zero_locs(em, z_thresh=.75):
    errors_gap = np.zeros(em.shape[2])
    em_slice_pix = em.shape[0] * em.shape[1]

    for i in range(em.shape[2]):
        if np.sum(np.logical_or(em[:,:,i] < .01*255, em[:,:,i] > .99*255)) > z_thresh * em_slice_pix:
            errors_gap[i] += 1
    return errors_gap

def correct_with_flow(em, warps, stride=25,scale=True, alternate=True, others_to_transform=[]):
    warped = em.copy()
    len_others = len(others_to_transform)
    others_warped = [x.copy() for x in others_to_transform]
    
    for w in warps.keys():
        warp_indices = list(range(w[0]+1, w[1]))
        rng = w[1] - w[0]
        for i in range(len(warp_indices)):
            if scale:
                if alternate:
                    alt = i % 2
                    warp_indx = w[alt]
                    alt = 1 if alt==0 else -1
                    indx = warp_indices[(i+1)//2 * alt]
                    scalar = alt#*((((i//2)+1)) / rng)

                else:
                    scalar = i+1
                    warp_indx = w[0]
                    indx = warp_indices[i]
            else:
                scalar = 1
                warp_indx = w[0]
                indx = warp_indices[i]
            warped_img = ndimage_warp(
                em[..., warp_indx], 
                warps[w]*scalar, 
                stride=(256,256), 
                work_size=(300,300), 
                overlap=(0, 0)
            )
            print("WARPED INDX", indx)
            warped[:, :, indx] = warped_img

            for j in range(len_others):
                warped_img = ndimage_warp(
                    others_to_transform[j][..., warp_indx], 
                    warps[w]*scalar, 
                    stride=(256,256), 
                    work_size=(300,300), 
                    overlap=(0, 0)
                )
                others_warped[j][:, :, indx] = warped_img
    return [warped, *others_warped]

def rigid_transform_slices(em, shifts, radius=None):
    if type(radius) == type(None):
        radius = [em.shape[0] // 2, em.shape[1] // 2]
        
    shifts = shifts.astype(int)
    em_center = np.array([em.shape[0]//2, em.shape[0]//2])
    
    em_new = np.zeros((radius[0]*2, radius[1]*2, em.shape[2]), dtype=np.float32)
    em_new_center = np.array([em_new.shape[0]//2, em_new.shape[0]//2])

    em_new[..., em.shape[2]//2] = em[em_center[0] - radius[0]:em_center[0] + radius[0],
                                     em_center[1] - radius[1]:em_center[1] + radius[1],
                                     em.shape[2]//2]

    for i in range(em.shape[2]//2+1, em.shape[2]):
        bound_1 = em_center[0] - radius[0] + shifts[i, 0]
        bound_2 = em_center[0] + radius[0] + shifts[i, 0]
        bound_3 = em_center[1] - radius[1] + shifts[i, 1]
        bound_4 = em_center[1] + radius[1] + shifts[i, 1]
        b1 = 0 if bound_1 >= 0 else abs(bound_1)
        b3 = 0 if bound_3 >= 0 else abs(bound_3)
        b2 = em_new.shape[0] if bound_2 <= em.shape[0] else em_new.shape[0] - abs(bound_2 - em.shape[0])
        b4 = em_new.shape[0] if bound_4 <= em.shape[0] else em_new.shape[0] - abs(bound_4 - em.shape[0])
        if b1 >= b2 or b3>=b4:
            continue
        em_new[b1:b2,b3:b4, i] = em[max([bound_1, 0]):min([bound_2, em.shape[0]]), 
                                    max([bound_3, 0]):min([bound_4, em.shape[0]]),
                                    i] 
        
    for i in range(em.shape[2]//2 - 1, -1, -1):
        bound_1 = em_center[0] - radius[0] - shifts[i, 0]
        bound_2 = em_center[0] + radius[0] - shifts[i, 0]
        bound_3 = em_center[1] - radius[1] - shifts[i, 1]
        bound_4 = em_center[1] + radius[1] - shifts[i, 1]

        b1 = 0 if bound_1 >= 0 else abs(bound_1)
        b3 = 0 if bound_3 >= 0 else abs(bound_3)
        b2 = em_new.shape[0] if bound_2 <= em.shape[0] else em_new.shape[0] - abs(bound_2 - em.shape[0])
        b4 = em_new.shape[0] if bound_4 <= em.shape[0] else em_new.shape[0] - abs(bound_4 - em.shape[0])
        if b1 >= b2 or b3>=b4:
            continue
        em_new[b1:b2,b3:b4, i] = em[max([bound_1, 0]):min([bound_2, em.shape[0]]), 
                                    max([bound_3, 0]):min([bound_4, em.shape[0]]),
                                    i]
    return em_new

def sofima_stitch(em,  warps, patch_size=256, stride=128, warp_scale=1.1, others_to_transform = [], direction=0):
    maskcorr = JAXMaskedXCorrWithStatsCalculator()
#     patch_size = 150
#     stride = 75
    config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                                    num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                    dt_max=100, prefer_orig_order=True,
                                    start_cap=0.1, final_cap=10)
    kwargs = {"min_peak_ratio": 1.1, "min_peak_sharpness": 1.1, "max_deviation": 0, "max_magnitude": 0}
    zero_inds = {w[1]:w[0] for w in warps.keys()}
    pad_y = patch_size // 2 // stride
    pad_x = patch_size // 2 // stride
    warped_mat = em.copy()
    len_others = len(others_to_transform)
    others_warped = [x.copy() for x in others_to_transform]
    for i in range(len_others):
        mid_frame = others_to_transform[i].shape[2]//2
        others_warped[i][:, :, mid_frame] = others_to_transform[i][:, :, mid_frame]
        
    warped_mat[:, :, em.shape[2]//2] = em[:, :, em.shape[2]//2]
    if direction == 0:
        ranges = [range(em.shape[2]//2, 1, -1), range(em.shape[2]//2, em.shape[2]-1)]
        inds = [-1, 1]
    elif direction == -1:
        ranges = [range(em.shape[2]-1, 0, -1)]
        inds = [direction]
    elif direction == 1:
        ranges = [range(0, em.shape[2]-1)]
        inds = [direction]

    else:
        ranges = []

    for ind, sign in enumerate(inds):
        x = None
        skip = False
        continue_ind = -1
        for i in ranges[ind]:
            # print("sofima ind", sign, i, i+sign, warp_scale)
            tile_1 = warped_mat[:,:,i]
            tile_2 = em[:,:,i + sign]

            if i in zero_inds.keys():
                continue_ind = zero_inds[i]-sign

                skip=True
            if i == continue_ind:
                skip = False
            if not skip:
                # print(i, i+sign)
                flow_1 = maskcorr.flow_field(tile_2, tile_1, patch_size, stride)

                flow_pad_1 = np.pad(flow_1, [[0, 0], [pad_y, pad_y - 1], [pad_x, pad_x - 1]], constant_values=np.nan)
                ffield_clean_1 = clean_flow(flow_pad_1[:, np.newaxis, ...], **kwargs)

                x = np.zeros_like(ffield_clean_1)
                x, _, _ = np.array(mesh.relax_mesh(x, ffield_clean_1, config))
                x = np.array(x)[:,0,...]
            print("SADFASDFADF", skip, i, i+sign, continue_ind, np.mean(x))
            if x is not None:
                warped_img = ndimage_warp(
                    tile_2, 
                    warp_scale*x, 
                    stride=(256,256), 
                    work_size=(300,300), 
                    overlap=(0, 0)
                )
                warped_mat[..., i+sign] = warped_img

                for j in range(len_others):
                    warped_other = ndimage_warp(
                        others_to_transform[j][:, :, i+sign], 
                        warp_scale*x, 
                        stride=(256,256), 
                        work_size=(300,300), 
                        overlap=(0, 0)
                    )
                    others_warped[j][:, :, i+sign] = warped_other

    return [warped_mat, *others_warped]

def combined_stitch(em, errors, direction, patch_size = 160, stride = 40, alternate=True, others_to_transform=None):
    from sofima.flow_field import JAXMaskedXCorrWithStatsCalculator
    from sofima.flow_utils import clean_flow
    from sofima.warp import ndimage_warp
    from sofima import mesh
    from sofima import map_utils
    from connectomics.common import bounding_box
    from sofima import flow_utils
    others_to_transform = [] if others_to_transform is None else others_to_transform
    maskcorr = JAXMaskedXCorrWithStatsCalculator()

    config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                                    num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                    dt_max=100, prefer_orig_order=True,
                                    start_cap=0.1, final_cap=10)
    
    kwargs = {"min_peak_ratio": 1.1, "min_peak_sharpness": 1.1, "max_deviation": 0, "max_magnitude": 0}
    err_pres = []
    err_posts = []

    i=0
    flag = False

    while i < errors.shape[0]:
        if errors[i] and flag:
            pre = i-1
            while True:
                i+=1
                if i == errors.shape[0]:
                    break
                if not errors[i]:
                    post = i
                    err_pres.append(pre)
                    err_posts.append(post)
                    break
        elif not errors[i]:
            flag = True
        i+=1
    err_pres = np.array(err_pres)
    err_posts = np.array(err_posts)

    config = mesh.IntegrationConfig(dt=0.001, gamma=0., k0=0.01, k=0.1, stride=stride,
                                    num_iters=1000, max_iters=20000, stop_v_max=0.001,
                                    dt_max=100, prefer_orig_order=True,
                                    start_cap=0.1, final_cap=10)
    kwargs = {"min_peak_ratio": 1.1, "min_peak_sharpness": 1.1, "max_deviation": 0, "max_magnitude": 0}

    pad_y = patch_size // 2 // stride
    pad_x = patch_size // 2 // stride
    warped_mat = em.copy()
    warp_indices = []

    len_others = len(others_to_transform)
    others_warped = [x.copy() for x in others_to_transform]

    if direction == 0:
        ranges = [range(em.shape[2]//2, 1, -1), range(em.shape[2]//2, em.shape[2]-1)]
        inds = [-1, 1]
    elif direction == -1:
        ranges = [range(em.shape[2]-1, 1, -1)]
        inds = [direction]
    elif direction == 1:
        ranges = [range(0, em.shape[2]-1)]
        inds = [direction]
    else:
        ranges = []
    ct = 0
    for ind, sign in enumerate(inds):
        x = None
        dropped = False

        loc = -1
        for i in ranges[ind]:

            if i == loc+1:
                dropped=False
            if not dropped:
                if i in err_posts and direction == -1:
                    loc = int(err_pres[np.argwhere(err_posts == i)])
                    tile_1_idx = i
                    tile_2_idx =loc
                    w = [loc, i]
                    dropped = True

                elif i in err_pres and direction > -1:
                    loc = int(err_posts[np.argwhere(err_pres == i)])
                    tile_2_idx = loc
                    tile_1_idx = i
                    w = [i, loc]
                    dropped = True

                elif not dropped: 
                    tile_1_idx = i
                    tile_2_idx = i + sign

                if dropped:

                    drop_ct = 0
                    rng = w[1] - w[0]
                    warp_indices = list(range(w[0]+1, w[1]))
                else:
                    tile_1 = np.squeeze(em[:, :, tile_1_idx])
                    tile_2 = np.squeeze(em[:, :, tile_2_idx])

                    flow_1 = maskcorr.flow_field(tile_2, tile_1, patch_size, stride)

                    flow_pad_1 = np.pad(flow_1, [[0, 0], [pad_y, pad_y - 1], [pad_x, pad_x - 1]], constant_values=np.nan)
                    ffield_clean_1 = clean_flow(flow_pad_1[:, np.newaxis, ...], **kwargs)

                    tile_1_ds = tile_1.reshape(-1, 2, tile_1.shape[0]//2, 2).sum((-1, -3)) / 2
                    tile_2_ds = tile_2.reshape(-1, 2, tile_2.shape[0]//2, 2).sum((-1, -3)) / 2

                    flow_1 = maskcorr.flow_field(tile_2_ds, tile_1_ds, patch_size, stride)

                    flow_pad_1 = np.pad(flow_1, [[0, 0], [pad_y, pad_y - 1], [pad_x, pad_x - 1]], constant_values=np.nan)
                    ffield_clean_2 = clean_flow(flow_pad_1[:, np.newaxis, ...], **kwargs)

                    box1x = bounding_box.BoundingBox(start=(0, 0, 0), size=(ffield_clean_1.shape[-1], ffield_clean_1.shape[-1], 1))
                    box2x = bounding_box.BoundingBox(start=(0, 0, 0), size=(ffield_clean_2.shape[-1], ffield_clean_2.shape[-1], 1))

                    resampled = 2*map_utils.resample_map(
                        ffield_clean_2,  #
                        box2x, box1x, 1 / .5, 1)

                    ffield_com = flow_utils.reconcile_flows((resampled, ffield_clean_1), max_gradient=0, max_deviation=0, min_patch_size=0)

                    x = np.zeros_like(ffield_com)
                    x, _, _ = np.array(mesh.relax_mesh(x, ffield_com, config))
                    x = np.array(x)[:,0,...]

                ct += 1

            if x is not None:
                if dropped:
                    warp_idx = tile_2_idx
                    indx = i+sign#warp_indices[w_ind]
                    s = stride
                    scalar = 1 - ((drop_ct)) / (rng)
                    drop_ct += 1
                else:
                    warp_idx = tile_2_idx
                    s = stride
                    scalar=1
                    indx = i+sign
                print(tile_1_idx, warp_idx, indx, dropped, scalar, np.mean(x*scalar))
                if warp_idx > warped_mat.shape[2]:
                    warp_idx =  warped_mat.shape[2] - 1
                warped_img = ndimage_warp(
                    np.squeeze(warped_mat[:, :, warp_idx]),
                    x*scalar, 
                    stride=(s,s), 
                    work_size=(300,300), 
                    overlap=(0, 0)
                )
                warped_mat[..., indx] = warped_img

                for j in range(len_others):
                    warped_other = ndimage_warp(
                        np.squeeze(others_warped[j][:, :, warp_idx]), 
                        x*scalar, 
                        stride=(s,s), 
                        work_size=(300,300), 
                        overlap=(0, 0)
                    )
                    others_warped[j][:, :, indx] = warped_other
        return [warped_mat, *others_warped]